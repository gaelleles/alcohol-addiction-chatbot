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
4. **Indexing** — the vectors are stored in a local **ChromaDB** vector store, persisted to disk under `.chroma_cache/` so this step doesn't have to be repeated on every restart (see [Speeding up local testing](#speeding-up-local-testing)).
5. **Retrieval** — when a question comes in, it's embedded with the same model and matched against the index to pull the top-k most similar chunks.
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
- `PRIVACY` — either `public` or `private`, **lowercase** (see [Privacy / IP](#privacy--ip) below). Any other value (including different casing, e.g. `PUBLIC`) silently falls through and the app will try to load PDFs directly from `pdf_papers/`, which is empty — you'll get a "no text could be extracted" error.

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

The vector index built in step 4 above is persisted to `.chroma_cache/<PRIVACY>/`. On subsequent runs, if a cache already exists for the current `PRIVACY` value, the app loads it directly instead of re-parsing and re-embedding every PDF — so the slow indexing step only happens once per corpus, not on every restart.

If you change the contents of `pdf_papers/` and want the index rebuilt, delete the relevant cache folder before restarting:

```bash
rm -rf .chroma_cache/public   # or .chroma_cache/private
```

For quick iteration on the app itself (not the retrieval quality), testing with `PRIVACY=public` is by far the fastest option, since the corpus is tiny.

# Evaluation

Getting an answer back from the chatbot doesn't tell you whether that answer is any good — RAG can silently fail in two independent places: retrieval can pull the wrong passages, or the LLM can generate an answer not actually supported by the passages it was given (hallucination), even when retrieval worked fine. [`eval/evaluate.py`](eval/evaluate.py) runs the real `rag_chain` (the same code the app uses, via `rag_pipeline.py`) against a small hand-written gold question set ([`eval/testset.py`](eval/testset.py)) grounded in `pdf_papers/public/`, and scores it with four [RAGAS](https://github.com/explodinggradients/ragas) metrics:

| Metric | Answers |
|---|---|
| `faithfulness` | Does the answer only assert things actually supported by the retrieved context? (catches hallucination) |
| `answer_relevancy` | Does the answer actually address the question asked? |
| `context_precision` | Is the retrieved context relevant to the question, with the most useful chunks ranked first? |
| `context_recall` | Did retrieval surface the information needed to produce a correct answer? |

Run it with:

```bash
uv run python eval/evaluate.py
```

## Judge model

RAGAS needs an LLM to *judge* the RAG chain's answers — a separate role from the chat model being evaluated. It's tempting to reuse the same Groq model the app uses (`openai/gpt-oss-20b`), but that model turned out to be an unreliable judge: `gpt-oss` is a reasoning model, and on RAGAS' stricter structured-output schemas it would intermittently leak its chain-of-thought into the response instead of emitting a clean tool call, failing anywhere from 1 in 3 to 3 in 3 attempts depending on the metric. `qwen/qwen3.8-27b` was tested and found reliable (identical scores across repeated runs of all four metrics), so it's used as the judge instead — a good general lesson: **the judge model matters as much as the model being evaluated, and needs its own reliability check**, not just a hope that "any chat model" can fill that role. See the comments in `eval/evaluate.py::build_judge_llm` for the exact `instructor` mode this required.

## Known constraints of running this on a free tier

- **Output-token budget**: some longer, more information-dense answers require the judge to decompose many atomic statements (each with a verdict and reasoning), which can exceed this Groq account's free-tier output-tokens-per-minute limit for the judge model. When a metric can't be scored after a few retries, the harness records it as `n/a` and excludes it from the averages rather than crashing the whole run — see `score_with_retries` in `eval/evaluate.py`.
- **Transient network blocks**: this project occasionally hits a `403 Access denied` from Groq that clears up on its own after a while (see the "Tech challenges" section for the same issue affecting the app itself). If a run fails outright with this error, it's Groq/network, not the harness — just retry later.

## Sample output

From a real run against the public corpus (your own numbers will vary slightly, since the judge is itself an LLM):

```
Q: How much higher is the rate of alcoholism among Native Americans compared to the U.S. average?
  faithfulness       1.000
  answer_relevancy   0.926
  context_precision  1.000
  context_recall     0.500

Q: Why does women's drinking warrant serious attention from researchers even though women consume less alcohol than men?
  faithfulness       0.400
  answer_relevancy   0.798
  context_precision  1.000
  context_recall     1.000
```

That `context_recall: 0.500` and `faithfulness: 0.400` are genuinely useful signal, not just numbers to wave at — they're pointing at real, inspectable weaknesses: a `context_recall` below 1.0 means the retriever didn't surface every fact needed to fully answer that question (a chunking/retrieval-tuning problem), while a low `faithfulness` means the LLM asserted something in its answer that the retrieved chunks didn't actually support (a prompting/hallucination problem) — and `eval/results.json` (written after each run) has the full per-statement judge reasoning behind every score, so you can go read exactly which claim failed and why instead of guessing.

# Tech challenges

I initially used FAISS, but I ran into a segmentation/indexing issue with Python 3.13 and NumPy 2.0. So I had to debug LangChain's abstraction layer to either implement manual injection via `from_texts` or switch to ChromaDB to ensure the system's stability.

Encrypted PDFs (AES) require the `cryptography` package for `pypdf` to be able to decrypt and extract their text — if you see `pypdf.errors.DependencyError: cryptography>=3.1 is required for AES algorithm`, make sure dependencies are installed via `uv sync` (this is already declared in `pyproject.toml`).

Occasionally Groq returns `groq.PermissionDeniedError: Error code: 403 - Access denied. Please check your network settings.` for every request, including ones that authenticate and work fine minutes later. This looks like a transient, IP-based restriction on Groq's side rather than anything wrong with the API key or the code — if you hit it, the API key reaches Groq fine (confirmed by getting an actual error back instead of an auth failure), so just wait and retry rather than re-checking `.env`.
