import os

import streamlit as st
from dotenv import load_dotenv

import rag_pipeline

load_dotenv()

st.set_page_config(page_title="Research Assistant", layout="wide")

# --- CONFIGURATION ---
API_SERVICE_KEY = os.getenv("API_SERVICE_KEY")
if not API_SERVICE_KEY:
    st.error(
        "Error: the API key was not found in the .env file. "
        "Set it up as described in the README."
    )
    st.stop()

try:
    PRIVACY = rag_pipeline.normalize_privacy(os.getenv("PRIVACY"))
except ValueError as e:
    st.error(f"Configuration error: {e}")
    st.stop()

PDF_DIRECTORY = rag_pipeline.get_pdf_directory(PRIVACY)
CHROMA_PERSIST_DIRECTORY = rag_pipeline.get_persist_directory(PRIVACY)


@st.cache_resource
def init_rag_chain():
    embeddings = rag_pipeline.build_embeddings()
    vectorstore = rag_pipeline.build_vectorstore(
        PDF_DIRECTORY, CHROMA_PERSIST_DIRECTORY, embeddings
    )
    llm = rag_pipeline.build_llm(API_SERVICE_KEY)
    return rag_pipeline.build_rag_chain(vectorstore, llm)


st.title("📚 Research Assistant")

if not os.path.exists(PDF_DIRECTORY):
    st.error(f"Can't find folder '{PDF_DIRECTORY}'.")
else:
    try:
        rag_chain = init_rag_chain()
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("sources"):
                st.markdown("**📍 Files accessed for this answer:**")
                for s in message["sources"]:
                    st.caption(s)

    query = st.chat_input("Ask your question:")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching the literature..."):
                result = rag_chain.invoke({"input": query})
            answer = result["answer"]
            st.write(answer)

            sources = rag_pipeline.format_sources(result["context"])

            if sources:
                st.markdown("**📍 Files accessed for this answer:**")
                for s in sources:
                    st.caption(s)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
