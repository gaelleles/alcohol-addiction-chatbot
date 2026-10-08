"""Building blocks of the RAG pipeline, shared by app.py and eval/evaluate.py."""

import hashlib
import os
import shutil

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHAT_MODEL_NAME = "openai/gpt-oss-20b"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVAL_K = 4
VALID_PRIVACY_VALUES = ("public", "private")
FINGERPRINT_FILENAME = "pdf_fingerprint.txt"

SYSTEM_PROMPT = (
    "You are a rigorous scientific research assistant. "
    "Make use of the following extracts to answer the question. "
    "If you don't know, say you don't know. "
    "At the end of your answer, list the sources (filename and page) you used. "
    "\n\n"
    "{context}"
)


def normalize_privacy(privacy: str | None) -> str:
    value = (privacy or "").strip().lower()
    if value not in VALID_PRIVACY_VALUES:
        raise ValueError(
            f"PRIVACY must be one of {VALID_PRIVACY_VALUES} (got {privacy!r}). "
            "Set it in your .env file, see the README."
        )
    return value


def get_pdf_directory(privacy: str | None) -> str:
    return f"./pdf_papers/{normalize_privacy(privacy)}/"


def get_persist_directory(
    privacy: str | None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> str:
    # One cache per chunking config, so comparing configs never mixes indexes.
    privacy = normalize_privacy(privacy)
    return f"./.chroma_cache/{privacy}/chunk{chunk_size}_overlap{chunk_overlap}"


def compute_pdf_fingerprint(
    pdf_directory: str, embedding_model: str = EMBEDDING_MODEL_NAME
) -> str:
    """Cheap hash of what the index depends on: PDF names, sizes, mtimes, model.

    mtime can change without the content changing (e.g. a re-download); that
    only costs an unnecessary rebuild, never a stale index.
    """
    digest = hashlib.sha256(embedding_model.encode())
    for name in sorted(os.listdir(pdf_directory)):
        if not name.endswith(".pdf"):
            continue
        stat = os.stat(os.path.join(pdf_directory, name))
        digest.update(f"{name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def is_cache_fresh(persist_directory: str, fingerprint: str) -> bool:
    path = os.path.join(persist_directory, FINGERPRINT_FILENAME)
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        return f.read().strip() == fingerprint


def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_vectorstore(
    pdf_directory: str,
    persist_directory: str,
    embeddings: HuggingFaceEmbeddings,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Chroma:
    fingerprint = compute_pdf_fingerprint(pdf_directory)
    if is_cache_fresh(persist_directory, fingerprint):
        return Chroma(
            persist_directory=persist_directory, embedding_function=embeddings
        )

    # Missing or stale cache (PDFs added, removed or modified): rebuild it.
    shutil.rmtree(persist_directory, ignore_errors=True)

    loader = DirectoryLoader(pdf_directory, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)

    if len(docs) == 0:
        raise ValueError(
            f"PDFs loaded from '{pdf_directory}' but no text could be extracted."
        )

    vectorstore = Chroma.from_documents(
        docs, embeddings, persist_directory=persist_directory
    )
    with open(os.path.join(persist_directory, FINGERPRINT_FILENAME), "w") as f:
        f.write(fingerprint)
    return vectorstore


def build_llm(api_key: str) -> ChatGroq:
    return ChatGroq(groq_api_key=api_key, model_name=CHAT_MODEL_NAME)


def build_rag_chain(vectorstore: Chroma, llm: ChatGroq, k: int = RETRIEVAL_K):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return create_retrieval_chain(retriever, question_answer_chain)


def format_sources(context_docs) -> list[str]:
    sources = set()
    for doc in context_docs:
        source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page")
        if isinstance(page, int):
            # PyPDFLoader pages are 0-indexed.
            sources.add(f"📄 {source_name} (Page {page + 1})")
        else:
            sources.add(f"📄 {source_name}")
    return sorted(sources)
