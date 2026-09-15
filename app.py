import streamlit as st
import os
from dotenv import load_dotenv

import rag_pipeline

load_dotenv()

# --- CONFIGURATION ---
PRIVACY = os.getenv("PRIVACY")
API_SERVICE_KEY = os.getenv("API_SERVICE_KEY")
PDF_DIRECTORY = rag_pipeline.get_pdf_directory(PRIVACY)
CHROMA_PERSIST_DIRECTORY = rag_pipeline.get_persist_directory(PRIVACY)

st.set_page_config(page_title="DS Research Assistant", layout="wide")

if not API_SERVICE_KEY:
    st.error("Erreur : La clé API n'a pas été trouvée dans le fichier .env. Configurez-la selon le README.")
    st.stop()

@st.cache_resource
def init_knowledge_base():
    embeddings = rag_pipeline.build_embeddings()
    return rag_pipeline.build_vectorstore(PDF_DIRECTORY, CHROMA_PERSIST_DIRECTORY, embeddings)

st.title("📚 Research Assistant")

if not os.path.exists(PDF_DIRECTORY):
    st.error(f"Can't find folder '{PDF_DIRECTORY}'.")
else:
    try:
        vectorstore = init_knowledge_base()
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()

    llm = rag_pipeline.build_llm(API_SERVICE_KEY)
    rag_chain = rag_pipeline.build_rag_chain(vectorstore, llm)

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

            sources_utilisees = rag_pipeline.format_sources(result["context"])

            if sources_utilisees:
                st.markdown("**📍 Files accessed for this answer:**")
                for s in sources_utilisees:
                    st.caption(s)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources_utilisees,
        })
