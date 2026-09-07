# Alcohol Addiction Chatbot

RAG+Streamlit chatbot specialised on addiction and alcohol issues.
This serves as a helper for a research project on addiction and alcohol-related issues.

Ask a question in natural language, and the chatbot answers using only the content of a curated library of PDFs (academic papers, books, reports), citing the exact file and page it drew the answer from.

# How it works

This is a **Retrieval-Augmented Generation (RAG)** pipeline: instead of asking a general-purpose LLM to answer from its own training data (which knows nothing about this specific, non-public paper collection and would be prone to hallucinating), the app retrieves the most relevant passages from the actual documents and asks the LLM to answer strictly from that retrieved context, citing sources.

The pipeline, step by step (see `app.py`):

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

Then update `model_name="..."` in `app.py` (where `ChatGroq` is instantiated) to a chat-capable model from that list.

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

# Tech challenges

I initially used FAISS, but I ran into a segmentation/indexing issue with Python 3.13 and NumPy 2.0. So I had to debug LangChain's abstraction layer to either implement manual injection via `from_texts` or switch to ChromaDB to ensure the system's stability.

Encrypted PDFs (AES) require the `cryptography` package for `pypdf` to be able to decrypt and extract their text — if you see `pypdf.errors.DependencyError: cryptography>=3.1 is required for AES algorithm`, make sure dependencies are installed via `uv sync` (this is already declared in `pyproject.toml`).
