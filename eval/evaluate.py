"""RAGAS evaluation harness for the RAG pipeline.

Runs the real rag_chain (see rag_pipeline.py) against the gold question set
(eval/testset.py, grounded in pdf_papers/public/) and reports two kinds of
metrics.

LLM-free retrieval metrics, computed for every run:

- page_hit:   was a chunk from the expected PDF page among the retrieved ones?
- source_hit: was a chunk from the expected PDF among the retrieved ones?
- context_chars: average size of the retrieved context (a cost proxy).

RAGAS metrics, scored by an LLM judge (skipped with --retrieval-only):

- faithfulness:       does the answer only assert things supported by the
                       retrieved context? (catches hallucination)
- answer_relevancy:   does the answer actually address the question asked?
- context_precision:  is the retrieved context relevant to the question,
                       ranked with the most useful chunks first?
- context_recall:     did retrieval surface the information needed to
                       produce the reference answer?

The judge is a different Groq-hosted model than the one being evaluated (see
DEFAULT_JUDGE_MODEL below (--judge-model)), reached through Groq's OpenAI-compatible endpoint
so no separate OpenAI key is needed. Embeddings reuse the same local
sentence-transformer as the app.

Usage:
    uv run python eval/evaluate.py                       # baseline (500/50, k=4)
    uv run python eval/evaluate.py --chunk-size 1000 --k 4
    uv run python eval/evaluate.py --retrieval-only --k 8   # no LLM, no API key
    uv run python eval/compare.py eval/results/*.json    # before/after table
"""

import argparse
import asyncio
import json
import os
import re
import sys

import instructor
import openai
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rag_pipeline  # noqa: E402

from ragas.llms.base import InstructorLLM, InstructorModelArgs  # noqa: E402
from ragas.embeddings.huggingface_provider import (  # noqa: E402
    HuggingFaceEmbeddings as RagasHuggingFaceEmbeddings,
)
from ragas.metrics.collections import (  # noqa: E402
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from testset import TESTSET  # noqa: E402

GROQ_OPENAI_COMPATIBLE_BASE_URL = "https://api.groq.com/openai/v1"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# Default judge model, deliberately different from the one being evaluated
# (rag_pipeline.CHAT_MODEL_NAME, openai/gpt-oss-20b). Both openai/gpt-oss-20b
# and openai/gpt-oss-120b were tried first and are unreliable at the strict
# tool-calling RAGAS needs for structured judge output on this Groq account:
# gpt-oss is a reasoning model and intermittently leaks its chain-of-thought
# into the response instead of emitting a clean tool call, failing anywhere
# from 1/3 to 3/3 attempts depending on the metric's schema complexity.
# qwen/qwen3.8-27b was verified reliable (identical scores across 3 repeated
# runs of all four metrics) and is used as the judge instead.
DEFAULT_JUDGE_MODEL = "qwen/qwen3.8-27b"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chunk-size", type=int, default=rag_pipeline.CHUNK_SIZE)
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="default: 10%% of the chunk size (the baseline's 500/50 ratio)",
    )
    parser.add_argument("--k", type=int, default=rag_pipeline.RETRIEVAL_K)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="skip generation and judging: no LLM call, no API key needed",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(RAGAS_METRICS),
        help="comma-separated RAGAS metrics to compute (default: all four). "
        "Judge calls are billed against a free-tier daily token quota: "
        "context_precision,context_recall cost about 4k tokens per question, "
        "faithfulness and answer_relevancy about 15k.",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--label", default=None, help="name of the results file")
    parser.add_argument("--limit", type=int, default=None, help="first N questions")
    args = parser.parse_args()
    args.metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = set(args.metrics) - set(RAGAS_METRICS)
    if unknown:
        parser.error(f"unknown metrics: {sorted(unknown)}")
    if args.chunk_overlap is None:
        args.chunk_overlap = args.chunk_size // 10
    if args.label is None:
        suffix = "_retrieval" if args.retrieval_only else ""
        args.label = f"chunk{args.chunk_size}_k{args.k}{suffix}"
    return args


def build_vectorstore(args):
    # Always evaluate against the public corpus, regardless of the local
    # PRIVACY setting, so results are reproducible by anyone cloning the repo.
    embeddings = rag_pipeline.build_embeddings()
    return rag_pipeline.build_vectorstore(
        rag_pipeline.get_pdf_directory("public"),
        rag_pipeline.get_persist_directory(
            "public", args.chunk_size, args.chunk_overlap
        ),
        embeddings,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


def retrieval_metrics(case, docs):
    retrieved = [
        {
            "file": os.path.basename(doc.metadata.get("source", "")),
            "page": doc.metadata.get("page", -1) + 1,  # 0-indexed -> 1-indexed
        }
        for doc in docs
    ]
    return {
        "retrieved": retrieved,
        "source_hit": any(r["file"] == case["source"] for r in retrieved),
        "page_hit": any(
            r["file"] == case["source"] and r["page"] in case["pages"]
            for r in retrieved
        ),
        "context_chars": sum(len(doc.page_content) for doc in docs),
    }


def run_testset(cases, vectorstore, args, api_key):
    """Retrieval (and generation, unless --retrieval-only) for every question."""
    rag_chain = None
    if not args.retrieval_only:
        llm = rag_pipeline.build_llm(api_key)
        rag_chain = rag_pipeline.build_rag_chain(vectorstore, llm, k=args.k)

    samples = []
    for i, case in enumerate(cases, start=1):
        print(f"Question {i}/{len(cases)}: {case['question']}")
        sample = {"question": case["question"], "reference": case["reference"]}
        if rag_chain is None:
            docs = vectorstore.similarity_search(case["question"], k=args.k)
            sample["answer"] = None
        else:
            try:
                result = rag_chain.invoke({"input": case["question"]})
                docs = result["context"]
                sample["answer"] = result["answer"]
            except Exception as e:  # one failed call must not lose the whole run
                print(f"  generation failed, RAGAS scores will be n/a: {e}")
                docs = vectorstore.similarity_search(case["question"], k=args.k)
                sample["answer"] = None
        sample["retrieved_contexts"] = [doc.page_content for doc in docs]
        sample.update(retrieval_metrics(case, docs))
        samples.append(sample)
    return samples


def build_judge_llm(api_key: str, model: str) -> InstructorLLM:
    judge_client = openai.AsyncOpenAI(
        api_key=api_key, base_url=GROQ_OPENAI_COMPATIBLE_BASE_URL
    )
    # ragas.llms.llm_factory(..., provider="openai") forces instructor.Mode.JSON
    # (OpenAI's json_object response format), which Groq's gpt-oss models fail
    # against ("Failed to validate JSON", empty completion). Plain Mode.TOOLS
    # is also unreliable here: on more complex schemas (e.g. ContextRecall's
    # per-statement classification list) the model sometimes wraps its answer
    # in a made-up tool call named "json" instead of the expected schema name.
    # Mode.TOOLS_STRICT (OpenAI-style strict function calling) was verified
    # reliable across repeated runs of all four metrics, so it's used here
    # instead of the ragas default.
    patched_client = instructor.from_openai(
        judge_client, mode=instructor.Mode.TOOLS_STRICT
    )
    return InstructorLLM(
        client=patched_client,
        model=model,
        provider="openai",
        # ragas' default max_tokens=1024 truncates mid-JSON for longer RAG
        # answers, once faithfulness needs to emit many atomic statements
        # with a verdict + reasoning each — observed as a "Failed to call a
        # function" error with an obviously cut-off failed_generation.
        # Can't raise this past ~1000 though: this Groq account's free tier
        # caps the qwen models at 1000 output-tokens-per-minute, and a
        # request whose max_tokens alone exceeds that budget is rejected
        # outright (429) before generation even starts.
        model_args=InstructorModelArgs(max_tokens=950),
    )


class DailyLimitReached(Exception):
    """The judge's daily token quota is spent; retrying cannot help."""


def retry_delay_seconds(message: str):
    """Parses Groq's 'Please try again in 1m2.5s' hint, if present."""
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", message)
    if not match:
        return None
    return int(match.group(1) or 0) * 60 + float(match.group(2))


async def score_with_retries(label: str, coro_factory):
    """Runs a metric's .ascore() call, handling judge failures without
    wasting the free-tier token quota.

    - Daily quota spent: raises DailyLimitReached (retrying only burns tokens).
    - Per-minute rate limit: waits for the delay Groq asks for, then retries.
    - Anything else (typically the judge failing to emit a clean structured
      output): one retry, since repeating a truncated generation rarely
      changes the outcome and every attempt is billed.

    Returns None (rather than raising) when a metric can't be scored, so one
    stubborn sample can't take down the whole run. The report shows these as
    "n/a" and excludes them from the averages.
    """
    other_failures = 0
    for attempt in range(1, 5):
        try:
            result = await coro_factory()
            return result.value
        except Exception as e:
            message = str(e)
            print(f"  [{label}] judge call failed (attempt {attempt}): {message[:200]}")
            if "model_not_found" in message or "invalid_api_key" in message:
                raise SystemExit(f"Judge misconfigured, not a scoring failure: {message}")
            if "tokens per day" in message:
                raise DailyLimitReached(message) from e
            if "rate_limit_exceeded" in message:
                delay = retry_delay_seconds(message)
                if delay is None or delay > 120:
                    break
                await asyncio.sleep(delay + 1)
                continue
            other_failures += 1
            if other_failures >= 2:
                break
    print(f"  [{label}] could not be scored, recording as n/a")
    return None


async def score_samples(samples, args, api_key: str):
    judge_llm = build_judge_llm(api_key, args.judge_model)

    # One factory per metric; only the selected ones are instantiated.
    metric_calls = {}
    if "faithfulness" in args.metrics:
        metric = Faithfulness(llm=judge_llm)
        metric_calls["faithfulness"] = lambda s: metric.ascore(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["retrieved_contexts"],
        )
    if "answer_relevancy" in args.metrics:
        judge_embeddings = RagasHuggingFaceEmbeddings(
            model=f"sentence-transformers/{rag_pipeline.EMBEDDING_MODEL_NAME}"
        )
        metric = AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings)
        metric_calls["answer_relevancy"] = lambda s: metric.ascore(
            user_input=s["question"], response=s["answer"]
        )
    if "context_precision" in args.metrics:
        metric = ContextPrecision(llm=judge_llm)
        metric_calls["context_precision"] = lambda s: metric.ascore(
            user_input=s["question"],
            reference=s["reference"],
            retrieved_contexts=s["retrieved_contexts"],
        )
    if "context_recall" in args.metrics:
        metric = ContextRecall(llm=judge_llm)
        metric_calls["context_recall"] = lambda s: metric.ascore(
            user_input=s["question"],
            retrieved_contexts=s["retrieved_contexts"],
            reference=s["reference"],
        )

    for sample in samples:
        for name in metric_calls:
            sample[name] = None

    # Sequential on purpose: Groq's free tier has per-minute rate limits.
    try:
        for i, sample in enumerate(samples, start=1):
            if sample["answer"] is None:
                continue
            print(f"Scoring question {i}/{len(samples)}...")
            for name, call in metric_calls.items():
                sample[name] = await score_with_retries(
                    name, lambda s=sample, c=call: c(s)
                )
    except DailyLimitReached as e:
        print(f"\nJudge daily token quota reached, stopping early: {e}")
        print("Remaining questions are recorded as n/a; rerun once the quota refills.")
    return samples


def summarize(samples, args):
    n = len(samples)
    summary = {
        "n_questions": n,
        "page_hit": sum(s["page_hit"] for s in samples) / n,
        "source_hit": sum(s["source_hit"] for s in samples) / n,
        "context_chars": sum(s["context_chars"] for s in samples) / n,
    }
    if not args.retrieval_only:
        for metric in args.metrics:
            scored = [s[metric] for s in samples if s.get(metric) is not None]
            summary[metric] = sum(scored) / len(scored) if scored else None
            summary[f"{metric}_n"] = len(scored)
    return summary


def print_report(summary, args):
    print("\n=== Summary ===")
    print(f"  questions          {summary['n_questions']}")
    print(f"  page_hit           {summary['page_hit']:.3f}")
    print(f"  source_hit         {summary['source_hit']:.3f}")
    print(f"  context_chars      {summary['context_chars']:.0f}")
    if args.retrieval_only:
        return
    for metric in args.metrics:
        value = summary[metric]
        shown = f"{value:.3f}" if value is not None else "n/a"
        print(f"  {metric:<18} {shown} (scored on {summary[metric + '_n']})")


def main():
    args = parse_args()
    load_dotenv()
    api_key = os.getenv("API_SERVICE_KEY")
    if not args.retrieval_only and not api_key:
        raise SystemExit(
            "API_SERVICE_KEY is not set: configure .env as described in the "
            "README, or use --retrieval-only."
        )

    cases = TESTSET[: args.limit] if args.limit else TESTSET
    print(
        f"[{args.label}] chunk_size={args.chunk_size} "
        f"chunk_overlap={args.chunk_overlap} k={args.k}"
    )
    vectorstore = build_vectorstore(args)
    samples = run_testset(cases, vectorstore, args, api_key)

    if not args.retrieval_only:
        print("Scoring answers with RAGAS...")
        samples = asyncio.run(score_samples(samples, args, api_key))

    summary = summarize(samples, args)
    print_report(summary, args)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{args.label}.json")
    config = {
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "k": args.k,
        "retrieval_only": args.retrieval_only,
        "embedding_model": rag_pipeline.EMBEDDING_MODEL_NAME,
        "chat_model": rag_pipeline.CHAT_MODEL_NAME,
        "judge_model": None if args.retrieval_only else args.judge_model,
        "metrics": [] if args.retrieval_only else args.metrics,
    }
    with open(path, "w") as f:
        json.dump(
            {"config": config, "summary": summary, "samples": samples}, f, indent=2
        )
    print(f"\nFull results written to {path}")


if __name__ == "__main__":
    main()
