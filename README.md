# Alcohol Addiction Chatbot

RAG+Streamlit chatbot specialised on addiction and alcohol issues.
This serves as a helper for a research project on addiction and alcohol-related issues.

Ask a question in natural language, and the chatbot answers using only the content of a curated library of PDFs (academic papers, books, reports), citing the exact file and page it drew the answer from.

# How it works

This is a **Retrieval-Augmented Generation (RAG)** pipeline: instead of asking a general-purpose LLM to answer from its own training data (which knows nothing about this specific, non-public paper collection and would be prone to hallucinating), the app retrieves the most relevant passages from the actual documents and asks the LLM to answer strictly from that retrieved context, citing sources.

The pipeline, step by step (see `rag_pipeline.py`, shared by the Streamlit app and the [evaluation harness](#evaluation)):

1. **Ingestion** — `DirectoryLoader` + `PyPDFLoader` (LangChain) read every PDF in the configured folder (see [Privacy](#privacy--ip) below), page by page.
2. **Chunking** — `RecursiveCharacterTextSplitter` splits the extracted text into chunks of **500 characters with 50 characters of overlap**. Smaller chunks make retrieval more precise (less irrelevant text gets pulled in alongside the relevant part); the overlap prevents an idea from being cleanly severed at a chunk boundary.
3. **Embedding** — each chunk is turned into a vector with a local embedding model (see [Models used](#models-used)), so no external API call or cost is needed for this step.
4. **Indexing** — the vectors are stored in a local **ChromaDB** vector store, persisted to disk under `.chroma_cache/` so this step doesn't have to be repeated on every restart. The cache is tied to a fingerprint of the PDFs (names, sizes, modification times) and is rebuilt automatically when they change (see [Speeding up local testing](#speeding-up-local-testing)).
5. **Retrieval** — when a question comes in, it's embedded with the same model and matched against the index to pull the top-k most similar chunks (`k=4`, LangChain's default, made explicit as `RETRIEVAL_K` in `rag_pipeline.py`).
6. **Generation** — the retrieved chunks are "stuffed" directly into the system prompt as `{context}`, and an LLM (see [Models used](#models-used)) answers the question using only that context, listing which sources it used.
7. **Source display** — each retrieved chunk carries metadata (source filename + page number), which the app deduplicates and displays under "Files accessed for this answer" so every answer is auditable against the literature.

The chat itself is a proper multi-turn interface (`st.chat_input` / `st.chat_message`, backed by `st.session_state`): the conversation persists as a scrollable history rather than resetting on every question. Note that each question is still retrieved independently — the retriever doesn't currently use prior turns as context, so a follow-up like "what about women specifically?" won't automatically inherit the topic of the previous question.

# Models used

| Purpose | Model | Where it runs | Why |
|---|---|---|---|
| Embeddings | [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (via `HuggingFaceEmbeddings` / `sentence-transformers`) | Locally, on CPU | Small, fast sentence-transformer (384-dim vectors) — no API cost, no external dependency for indexing the corpus. |
| Chat / generation | Currently `openai/gpt-oss-20b`, called through `ChatGroq` | [Groq](https://console.groq.com/) API | Groq's LPU hardware gives very low-latency inference and has a generous free tier — good fit for a research tool with no budget. |

**Important — Groq's hosted model lineup changes over time**, and which models are available depends on your account. Models get decommissioned (e.g. this project originally used `llama3-8b-8192`, which no longer exists), and not every model on Groq's docs is necessarily enabled for every key. If you see a `model_decommissioned` or `model_not_found` error from `groq`, check what your key currently has access to before guessing a replacement name:

```bash
curl -s https://api.groq.com/openai/v1/models \
  -H "Authorization: Bearer $API_SERVICE_KEY" | python3 -m json.tool
```

Then update `CHAT_MODEL_NAME` in `rag_pipeline.py` to a chat-capable model from that list.

# Setup

## Package manager

We use uv as the package and project manager for its speed. If you need to install it, refer to their [installation guide](https://docs.astral.sh/uv/getting-started/installation/).

Install the project's dependencies with:

```bash
uv sync
```

## .env and API key

Run `cp .env.example .env` and fill the necessary variables in `.env`:

- `API_SERVICE_KEY` — retrieve an API key from [GROQ](https://console.groq.com/) (the free tier is quite generous).
- `PRIVACY` — either `public` or `private` (case-insensitive, see [Privacy / IP](#privacy--ip) below). Any other value, or a missing one, stops the app with an explicit configuration error.

## Pre-commit

We use [pre-commit](https://pre-commit.com/#install) to ensure code quality.

## Privacy / IP

To protect the authors' intellectual property, I chose not to add the full list of papers to the public project. However, I left a sample of public papers that you can use to test the code. This is the `PRIVACY` setting in `.env`:

- `PRIVACY=public` → reads from `pdf_papers/public/` (a handful of openly-licensed papers, small and fast to index — use this for local development/testing).
- `PRIVACY=private` → reads from `pdf_papers/private/` (the full research library, gitignored/local-only). Feel free to change it to `private` if you build your own library of papers there.

I encourage you to build a library that is mindful of the intellectual property of authors.

## Run the project

```bash
uv run streamlit run app.py
```

This opens the app at `http://localhost:8501`. On first run, it downloads the embedding model and indexes every PDF in the configured folder — with `PRIVACY=private` (100+ documents) this can take several minutes; with `PRIVACY=public` (4 documents) it's a matter of seconds.

## Speeding up local testing

The vector index built in step 4 above is persisted to `.chroma_cache/<PRIVACY>/chunk<size>_overlap<overlap>/`, next to a fingerprint of the PDF folder (file names, sizes and modification times, plus the embedding model name). On subsequent runs, if the fingerprint still matches, the app loads the index directly instead of re-parsing and re-embedding every PDF — so the slow indexing step only happens once per corpus, not on every restart. If you add, remove or modify a PDF, the fingerprint changes and the index is rebuilt automatically.

The check is deliberately cheap (it doesn't hash file contents), so it can occasionally trigger a rebuild for a file that was only re-saved with the same content; it can't serve a stale index. To force a rebuild anyway:

```bash
rm -rf .chroma_cache/public   # or .chroma_cache/private
```

For quick iteration on the app itself (not the retrieval quality), testing with `PRIVACY=public` is by far the fastest option, since the corpus is tiny.

## Tests

```bash
uv run pytest
```

The unit tests cover the parts that can fail silently without an LLM: source formatting (including chunks with no page metadata), `PRIVACY` validation, and the cache-invalidation fingerprint.

# Evaluation

Getting an answer back from the chatbot doesn't tell you whether that answer is any good: RAG can fail silently in two independent places. Retrieval can pull the wrong passages, or the LLM can state things the retrieved passages don't support, even when retrieval worked. [`eval/evaluate.py`](eval/evaluate.py) runs the real `rag_chain` (the code the app uses, via `rag_pipeline.py`) against [`eval/testset.py`](eval/testset.py): 20 questions over the 4 public PDFs, each with a reference answer written from the source text and the PDF page it comes from.

It reports two layers of metrics:

**1. Retrieval metrics, no LLM involved** (deterministic, free, work with `--retrieval-only` and no API key):

| Metric | Meaning |
|---|---|
| `page_hit` | Share of questions for which a chunk from the expected PDF page was retrieved |
| `source_hit` | Same, at PDF level (weak with only 4 PDFs: it is easy to hit the right file) |
| `context_chars` | Average size of the retrieved context, a proxy for tokens sent to the LLM |

**2. [RAGAS](https://github.com/explodinggradients/ragas) metrics, scored by an LLM judge:**

| Metric | Answers |
|---|---|
| `faithfulness` | Does the answer only assert things supported by the retrieved context? (catches hallucination) |
| `answer_relevancy` | Does the answer actually address the question asked? |
| `context_precision` | Is the retrieved context relevant to the question, with the most useful chunks ranked first? |
| `context_recall` | Did retrieval surface the information needed to produce the reference answer? |

```bash
uv run python eval/evaluate.py                                   # baseline: 500/50 chunks, k=4, all four RAGAS metrics
uv run python eval/evaluate.py --chunk-size 250 --k 4            # another configuration
uv run python eval/evaluate.py --retrieval-only --k 8            # retrieval metrics only, no LLM, no API key
uv run python eval/evaluate.py --metrics context_precision,context_recall   # cheaper judge run
uv run python eval/compare.py eval/results/*.json                # before/after tables
```

Each run writes `eval/results/<label>.json` (gitignored): the configuration, a summary, and for each question the answer, the retrieved chunks and the scores. It does not store the judge's reasoning.

## Measured iteration: chunk size and k

The baseline uses 500-character chunks (50 overlap) and `k=4`. I swept three chunk sizes and two values of `k` with the retrieval metrics (20 questions, no LLM):

| chunk size / overlap | k | page hit | source hit | context chars |
|---|---|---|---|---|
| 250 / 25 | 4 | 0.95 | 1.00 | 847 |
| 250 / 25 | 8 | 0.95 | 1.00 | 1741 |
| **500 / 50 (baseline)** | **4** | **0.90** | **1.00** | **1821** |
| 500 / 50 | 8 | 0.95 | 1.00 | 3668 |
| 1000 / 100 | 4 | 0.90 | 1.00 | 3773 |
| 1000 / 100 | 8 | 0.95 | 1.00 | 7607 |

What this shows, and what it doesn't:

- **The gaps in `page_hit` are one question out of 20**, which is noise. I did not change the defaults on this evidence.
- **Cost is the clearer signal.** The context sent to the LLM grows with chunk size times `k`. 250/k=4 reaches the same `page_hit` as 500/k=8 with about a quarter of the context, and as the baseline with about half. This is the configuration to validate next.
- **`page_hit` is necessary, not sufficient.** It says the right page was retrieved, not that the retrieved chunk contains the answer. Checking that is the job of RAGAS `context_recall` and `context_precision`.
- **One question is missed by every configuration**: the share of bisexual and lesbian women with alcohol dependence, whose answer is in a table on page 4. Text extracted from a PDF table embeds badly, so changing chunk size or `k` doesn't help; it needs table-aware extraction.

## RAGAS results so far

I don't have a RAGAS before/after yet. What I have:

- **A partial baseline.** On the first full run (baseline configuration), the judge's daily quota ran out after 10 of the 20 questions. On those 10: `faithfulness` 0.81 (scored on only 6 of them), `answer_relevancy` 0.92, `context_precision` 0.69, `context_recall` 0.55. Treat these as indicative.
- **Why it stopped.** The judge model (below) has a free-tier quota of 200,000 tokens per day, and scoring all four metrics costs about 20,000 tokens per question, retries included. A full run on 20 questions therefore doesn't fit in one day, let alone a before/after pair. The quota appears to refill gradually rather than at once.
- **What the harness does about it now.** `--metrics` lets you score only `context_precision` and `context_recall`, which are the ones a retrieval change affects, for about 4,000 tokens per question. When the daily quota is exhausted the run stops cleanly and records the remaining questions as `n/a`; per-minute limits are waited out for the delay Groq asks for; a misconfigured judge model fails loudly instead of producing `n/a`. `eval/compare.py` compares configurations on the questions that were scored in every run, so judge failures don't bias the averages.

The next step is to run, on a day with a fresh quota:

```bash
uv run python eval/evaluate.py --metrics context_precision,context_recall
uv run python eval/evaluate.py --chunk-size 250 --metrics context_precision,context_recall
uv run python eval/compare.py eval/results/chunk500_k4.json eval/results/chunk250_k4.json
```

## Judge model

RAGAS needs an LLM to *judge* the RAG chain's answers, a separate role from the chat model being evaluated. Reusing the app's model (`openai/gpt-oss-20b`) fails: `gpt-oss` is a reasoning model, and on RAGAS' structured-output schemas it intermittently leaks its chain-of-thought into the response instead of emitting a clean tool call. `openai/gpt-oss-120b` behaves the same way: re-tested on the two simple retrieval metrics only, it failed about 80% of calls. `qwen/qwen3.8-27b` is the only model on this account that was reliable (identical scores across repeated runs), so it is the default judge (`--judge-model`). **The judge model matters as much as the model being evaluated, and needs its own reliability check.** See `eval/evaluate.py::build_judge_llm` for the exact `instructor` mode this required, and note that the model lineup on a Groq account changes over time (a second Qwen model that worked in September is gone).

Two more constraints of the free tier are handled in `eval/evaluate.py`: the output of a judge call is capped at about 1,000 tokens, which truncates `faithfulness` on long answers (those show up as `n/a`), and Groq occasionally answers every request with a `403` that clears up later (see "Tech challenges").

# Limitations and next steps

- **No conversational memory in retrieval.** Each question is retrieved on its own. The standard fix is a history-aware retriever that rewrites a follow-up ("and for women?") into a standalone question using the chat history before searching.
- **Basic retrieval.** Dense search only, with `all-MiniLM-L6-v2`, an English-centric model, while the private corpus also contains French books. Next steps would be a multilingual embedding model, hybrid (BM25 + dense) search and a reranker. The public gold set is English-only, so it can't measure the multilingual gain.
- **Tables are poorly retrieved.** The retrieval metrics show one question that no configuration answers: its answer sits in a table, and text extracted from a PDF table embeds badly. Table-aware extraction would be needed.
- **Small evaluation set.** 20 questions over 4 PDFs: a difference of one question is within noise, so the comparisons in the Evaluation section are indications, not proof.
- **Legacy LangChain chains.** `langchain_classic` chains were the quickest way to iterate. For multi-turn behaviour I would migrate to LCEL or LangGraph.
- **Generation is remote.** Embeddings and indexing run locally on CPU (and are cached), but the answer itself comes from a hosted model (Groq).
- **Streamlit only.** No HTTP API, CI, container or monitoring yet.

# Tech challenges

I initially used FAISS, but I ran into a segmentation/indexing issue with Python 3.13 and NumPy 2.0. So I had to debug LangChain's abstraction layer to either implement manual injection via `from_texts` or switch to ChromaDB to ensure the system's stability.

Encrypted PDFs (AES) require the `cryptography` package for `pypdf` to be able to decrypt and extract their text — if you see `pypdf.errors.DependencyError: cryptography>=3.1 is required for AES algorithm`, make sure dependencies are installed via `uv sync` (this is already declared in `pyproject.toml`).

Occasionally Groq returns `groq.PermissionDeniedError: Error code: 403 - Access denied. Please check your network settings.` for every request, including ones that authenticate and work fine minutes later. This looks like a transient, IP-based restriction on Groq's side rather than anything wrong with the API key or the code — if you hit it, the API key reaches Groq fine (confirmed by getting an actual error back instead of an auth failure), so just wait and retry rather than re-checking `.env`.
