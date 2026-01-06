import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --- CONFIGURATION ---
PRIVACY = os.getenv("PRIVACY")
API_SERVICE_KEY = os.getenv("API_SERVICE_KEY")
PDF_DIRECTORY = "./pdf_papers/"
if PRIVACY == "public":
    PDF_DIRECTORY += "public/"
if PRIVACY == "private":
    PDF_DIRECTORY += "private/"

st.set_page_config(page_title="DS Research Assistant", layout="wide")

if not API_SERVICE_KEY:
    st.error("Erreur : La clé API n'a pas été trouvée dans le fichier .env. Configurez-la selon le README.")
    st.stop()

@st.cache_resource
def init_knowledge_base():
    # Load PDFs
    loader = DirectoryLoader(PDF_DIRECTORY, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    # Reduced chunk size to locate information more precisely
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    if len(docs) == 0:
            st.error("❌ PDF loaded but no text could be extracted.")
            st.stop()
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore

st.title("📚 Research Assistant")

if not os.path.exists(PDF_DIRECTORY):
    st.error(f"Can't find folder '{PDF_DIRECTORY}'.")
else:
    vectorstore = init_knowledge_base()
    llm = ChatGroq(API_SERVICE_KEY=API_SERVICE_KEY, model_name="llama3-8b-8192")

    # Defining the system prompt for research
    system_prompt = (
        "You are a rigorous scientific research assistant. "
        "Make use of the following extracts to answer the question. "
        "If you don't know, say you don't know. "
        "At the end of your answer, list the sources (filename and page) you used. "
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 2. Création de la chaîne de traitement
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)

    query = st.text_input("Ask your question:")

    if query:
        result = rag_chain.invoke({"input": query})
        
        st.markdown("### Answer")
        st.write(result["answer"])
        
        st.markdown("### 📍 Files accessed for this answer:")
        # Retrieve metadata for accessed files
        sources_utilisees = []
        for doc in result["context"]:
            source_name = os.path.basename(doc.metadata.get('source', 'Inconnu'))
            page_num = doc.metadata.get('page', 'Inconnue')
            sources_utilisees.append(f"📄 {source_name} (Page {page_num + 1})") # +1 bc indexed from 0

        # Affichage unique des sources
        for s in sorted(list(set(sources_utilisees))):
            st.caption(s)
