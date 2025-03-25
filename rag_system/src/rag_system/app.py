import streamlit as st
from vecstore_manager import DocumentManager
from query_manager import ConversationalRetrievalAgent
import tempfile
import os
from pathlib import Path
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# App layout
st.set_page_config(page_title="RAG System", page_icon="📚", layout="wide")
st.markdown("""
    <style>
        :root {
            color-scheme: dark;
        }
        .main {
            background-color: #1e1e1e;
            padding: 20px;
        }
        .block-container {
            padding-top: 2rem;
        }
        h1, h2, h3, h4 {
            color: #fafafa;
        }
        .stTextInput > div > div > input {
            background-color: #2a2a2a;
            color: #f0f0f0;
        }
        .animate {
            animation: fadeIn 0.8s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
""", unsafe_allow_html=True)

st.title("📖 RAG Document Assistant")
st.markdown("Your personal assistant to upload documents, ask questions, and get answers — with full traceability. ✨")

# Sidebar: API Key
st.sidebar.header("🔐 API Configuration")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    with st.sidebar:
        st.markdown("<div class='animate'>✅ <strong>API Key set successfully!</strong></div>", unsafe_allow_html=True)
else:
    st.sidebar.warning("Please enter your OpenAI API key to proceed.")

# Upload documents section (now below API input)
st.subheader("📤 Upload Documents")
uploaded_files = st.file_uploader(
    "Choose documents:",
    accept_multiple_files=True,
    type=['md', 'txt', 'pdf', 'docx']
)

# Set paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = PROJECT_ROOT / "data" / "02_intermediate" / "vector_store"
CHROMA_PATH.mkdir(parents=True, exist_ok=True)

# Initialize session
if "agent" not in st.session_state:
    st.session_state.agent = None

if uploaded_files and api_key:
    with tempfile.TemporaryDirectory() as tmpdirname:
        docs_dir = os.path.join(tmpdirname, "uploaded_docs")
        os.makedirs(docs_dir, exist_ok=True)

        for uploaded_file in uploaded_files:
            file_path = os.path.join(docs_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        doc_manager = DocumentManager(
            directory_path=docs_dir,
            chroma_path=str(CHROMA_PATH),
            glob_pattern="*.*",
            embedding_model="openai"
        )

        st.info("Processing documents...")
        doc_manager.load_documents()
        sections = doc_manager.split_documents()
        doc_manager.create_chroma_store(sections)
        st.success(f"✅ {len(sections)} sections processed!")

        embeddings = OpenAIEmbeddings()
        vectordb = Chroma(
            persist_directory=str(CHROMA_PATH),
            embedding_function=embeddings
        )
        st.session_state.agent = ConversationalRetrievalAgent(vectordb)

elif uploaded_files and not api_key:
    st.warning("Please enter your OpenAI API key first!")

# Question and Answer section
st.subheader("💬 Ask a Question")
question = st.text_input("What would you like to know?")

if question:
    if st.session_state.agent:
        with st.spinner("Generating response..."):
            response = st.session_state.agent.ask_question(question)
            source_docs = st.session_state.agent.get_source_documents(question)

        st.markdown("---")
        st.markdown("### 📝 Answer")
        st.success(response)

        st.markdown("### 📚 Source Snippets")
        for i, doc in enumerate(source_docs, start=1):
            with st.expander(f"Source {i}"):
                st.write(doc.page_content)
    else:
        st.warning("Please upload documents and enter your API key first!")











