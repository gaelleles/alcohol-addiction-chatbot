"""Building blocks of the RAG pipeline, shared by the Streamlit app and eval/evaluate.py."""

import os

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHAT_MODEL_NAME = "openai/gpt-oss-20b"

SYSTEM_PROMPT = (
    "You are a rigorous scientific research assistant. "
    "Make use of the following extracts to answer the question. "
    "If you don't know, say you don't know. "
    "At the end of your answer, list the sources (filename and page) you used. "
    "\n\n"
    "{context}"
)


def get_pdf_directory(privacy: str) -> str:
    pdf_directory = "./pdf_papers/"
    if privacy == "public":
        pdf_directory += "public/"
    if privacy == "private":
        pdf_directory += "private/"
    return pdf_directory


def get_persist_directory(privacy: str) -> str:
    return f"./.chroma_cache/{privacy}"


def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_vectorstore(pdf_directory: str, persist_directory: str, embeddings: HuggingFaceEmbeddings) -> Chroma:
    # Reuse a previously built index instead of re-parsing and re-embedding
    # every PDF on each restart (slow with 100+ private documents).
    if os.path.isdir(persist_directory) and os.listdir(persist_directory):
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    loader = DirectoryLoader(pdf_directory, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Reduced chunk size to locate information more precisely
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    if len(docs) == 0:
        raise ValueError(f"PDFs loaded from '{pdf_directory}' but no text could be extracted.")

    return Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)


def build_llm(api_key: str) -> ChatGroq:
    return ChatGroq(groq_api_key=api_key, model_name=CHAT_MODEL_NAME)


def build_rag_chain(vectorstore: Chroma, llm: ChatGroq):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)


def format_sources(context_docs) -> list[str]:
    sources = []
    for doc in context_docs:
        source_name = os.path.basename(doc.metadata.get('source', 'Inconnu'))
        page_num = doc.metadata.get('page', 'Inconnue')
        sources.append(f"📄 {source_name} (Page {page_num + 1})")  # +1 bc indexed from 0
    return sorted(set(sources))
