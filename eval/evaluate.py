"""RAGAS evaluation harness for the RAG pipeline.

Runs the real rag_chain (see rag_pipeline.py) against a small gold question
set (eval/testset.py) grounded in pdf_papers/public/, then scores each
answer with four standard RAG metrics:

- faithfulness:       does the answer only assert things supported by the
                       retrieved context? (catches hallucination)
- answer_relevancy:   does the answer actually address the question asked?
- context_precision:  is the retrieved context relevant to the question,
                       ranked with the most useful chunks first?
- context_recall:     did retrieval surface the information needed to
                       produce the reference answer?

The judge is a larger Groq-hosted model than the one being evaluated (see
JUDGE_MODEL_NAME below), reached through Groq's OpenAI-compatible endpoint
so no separate OpenAI key is needed. Embeddings reuse the same local
sentence-transformer as the app.

Usage:
    uv run python eval/evaluate.py
"""

import asyncio
import json
import os
import sys

import instructor
import openai
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rag_pipeline  # noqa: E402

from ragas.llms.base import InstructorLLM, InstructorModelArgs
from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings as RagasHuggingFaceEmbeddings
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

from testset import TESTSET

GROQ_OPENAI_COMPATIBLE_BASE_URL = "https://api.groq.com/openai/v1"
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")

# Judge model, deliberately different from the one being evaluated
# (rag_pipeline.CHAT_MODEL_NAME, openai/gpt-oss-20b). Both openai/gpt-oss-20b
# and openai/gpt-oss-120b were tried first and are unreliable at the strict
# tool-calling RAGAS needs for structured judge output on this Groq account:
# gpt-oss is a reasoning model and intermittently leaks its chain-of-thought
# into the response instead of emitting a clean tool call, failing anywhere
# from 1/3 to 3/3 attempts depending on the metric's schema complexity.
# qwen/qwen3.8-27b was verified reliable (identical scores across 3 repeated
# runs of all four metrics) and is used as the judge instead.
JUDGE_MODEL_NAME = "qwen/qwen3.8-27b"


def build_rag_chain(api_key: str):
    # Always evaluate against the public corpus, regardless of the local
    # PRIVACY setting, so results are reproducible by anyone cloning the repo.
    pdf_directory = rag_pipeline.get_pdf_directory("public")
    persist_directory = rag_pipeline.get_persist_directory("public")

    embeddings = rag_pipeline.build_embeddings()
    vectorstore = rag_pipeline.build_vectorstore(pdf_directory, persist_directory, embeddings)
    llm = rag_pipeline.build_llm(api_key)
    return rag_pipeline.build_rag_chain(vectorstore, llm)


def run_rag_over_testset(rag_chain):
    """Runs each test question through the real RAG chain (synchronous, like the app)."""
    samples = []
    for case in TESTSET:
        result = rag_chain.invoke({"input": case["question"]})
        samples.append({
            "question": case["question"],
            "reference": case["reference"],
            "answer": result["answer"],
            "retrieved_contexts": [doc.page_content for doc in result["context"]],
        })
    return samples


def build_judge_llm(api_key: str) -> InstructorLLM:
    judge_client = openai.AsyncOpenAI(api_key=api_key, base_url=GROQ_OPENAI_COMPATIBLE_BASE_URL)
    # ragas.llms.llm_factory(..., provider="openai") forces instructor.Mode.JSON
    # (OpenAI's json_object response format), which Groq's gpt-oss models fail
    # against ("Failed to validate JSON", empty completion). Plain Mode.TOOLS
    # is also unreliable here: on more complex schemas (e.g. ContextRecall's
    # per-statement classification list) the model sometimes wraps its answer
    # in a made-up tool call named "json" instead of the expected schema name.
    # Mode.TOOLS_STRICT (OpenAI-style strict function calling) was verified
    # reliable across repeated runs of all four metrics, so it's used here
    # instead of the ragas default.
    patched_client = instructor.from_openai(judge_client, mode=instructor.Mode.TOOLS_STRICT)
    return InstructorLLM(
        client=patched_client,
        model=JUDGE_MODEL_NAME,
        provider="openai",
        # ragas' default max_tokens=1024 truncates mid-JSON for longer RAG
        # answers, once faithfulness needs to emit many atomic statements
        # with a verdict + reasoning each — observed as a "Failed to call a
        # function" error with an obviously cut-off failed_generation.
        # Can't raise this past ~1000 though: this Groq account's free tier
        # caps qwen/qwen3.8-27b at 1000 output-tokens-per-minute, and a
        # request whose max_tokens alone exceeds that budget is rejected
        # outright (429) before generation even starts.
        model_args=InstructorModelArgs(max_tokens=950),
    )


async def score_with_retries(label: str, coro_factory, retries: int = 3):
    """Runs a metric's .ascore() call, retrying on judge flakiness.

    The judge is itself an LLM: on more complex schemas it occasionally
    leaks its reasoning into the response instead of emitting a clean
    structured tool call, which surfaces as an instructor/Groq parsing
    error. A fresh sample on retry usually succeeds.

    Returns None (rather than raising) once retries are exhausted, so one
    stubborn sample — e.g. a long answer whose faithfulness check needs
    more output tokens than the judge's free-tier rate limit allows — can't
    take down the whole evaluation run. print_report() reports these as
    "n/a" and excludes them from the averages.
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            result = await coro_factory()
            return result.value
        except Exception as e:
            last_exc = e
            print(f"  [{label}] judge call failed on attempt {attempt}/{retries}: {e}")
            if attempt < retries:
                await asyncio.sleep(2)
    print(f"  [{label}] giving up after {retries} attempts, recording as n/a: {last_exc}")
    return None


async def score_samples(samples, api_key: str):
    judge_llm = build_judge_llm(api_key)
    judge_embeddings = RagasHuggingFaceEmbeddings(model=f"sentence-transformers/{rag_pipeline.EMBEDDING_MODEL_NAME}")

    faithfulness = Faithfulness(llm=judge_llm)
    answer_relevancy = AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings)
    context_precision = ContextPrecision(llm=judge_llm)
    context_recall = ContextRecall(llm=judge_llm)

    # Sequential on purpose: Groq's free tier has a per-minute rate limit,
    # and each sample already issues several judge calls (one per metric,
    # more for answer_relevancy which samples multiple candidate questions).
    for i, sample in enumerate(samples):
        print(f"Scoring question {i + 1}/{len(samples)}...")
        sample["faithfulness"] = await score_with_retries(
            "faithfulness",
            lambda s=sample: faithfulness.ascore(
                user_input=s["question"],
                response=s["answer"],
                retrieved_contexts=s["retrieved_contexts"],
            ),
        )
        sample["answer_relevancy"] = await score_with_retries(
            "answer_relevancy",
            lambda s=sample: answer_relevancy.ascore(
                user_input=s["question"],
                response=s["answer"],
            ),
        )
        sample["context_precision"] = await score_with_retries(
            "context_precision",
            lambda s=sample: context_precision.ascore(
                user_input=s["question"],
                reference=s["reference"],
                retrieved_contexts=s["retrieved_contexts"],
            ),
        )
        sample["context_recall"] = await score_with_retries(
            "context_recall",
            lambda s=sample: context_recall.ascore(
                user_input=s["question"],
                retrieved_contexts=s["retrieved_contexts"],
                reference=s["reference"],
            ),
        )
    return samples


def print_report(samples):
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    print("\n=== Per-question scores ===")
    for s in samples:
        print(f"\nQ: {s['question']}")
        for m in metrics:
            value = s[m]
            print(f"  {m:<18} {value:.3f}" if value is not None else f"  {m:<18} n/a")

    print("\n=== Averages across the testset ===")
    for m in metrics:
        scored = [s[m] for s in samples if s[m] is not None]
        skipped = len(samples) - len(scored)
        avg = sum(scored) / len(scored) if scored else float("nan")
        note = f" ({skipped} skipped)" if skipped else ""
        print(f"  {m:<18} {avg:.3f}{note}")


def main():
    load_dotenv()
    api_key = os.getenv("API_SERVICE_KEY")
    if not api_key:
        raise SystemExit("API_SERVICE_KEY is not set — configure .env as described in the README.")

    print(f"Running {len(TESTSET)} questions through the RAG chain...")
    rag_chain = build_rag_chain(api_key)
    samples = run_rag_over_testset(rag_chain)

    print("Scoring answers with RAGAS...")
    samples = asyncio.run(score_samples(samples, api_key))

    print_report(samples)

    with open(RESULTS_PATH, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
