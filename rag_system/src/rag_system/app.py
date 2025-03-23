import streamlit as st
from vecstore_manager import DocumentManager
import tempfile
import os

st.title("Simple RAG System")

# Setup paths
CHROMA_PATH = "chroma_db"

# Section for uploading documents
st.header("Upload Documents")
uploaded_files = st.file_uploader(
    "Choose documents (multiple allowed):",
    accept_multiple_files=True,
    type=['md', 'txt', 'pdf', 'docx']
)

if uploaded_files:
    with tempfile.TemporaryDirectory() as tmpdirname:
        docs_dir = os.path.join(tmpdirname, "uploaded_docs")
        os.makedirs(docs_dir, exist_ok=True)

        # Save uploaded files temporarily
        for uploaded_file in uploaded_files:
            file_path = os.path.join(docs_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        # Initialize DocumentManager
        doc_manager = DocumentManager(
            directory_path=docs_dir,
            chroma_path=CHROMA_PATH,
            glob_pattern="*.*",
            embedding_model="openai"
        )

        st.info("Loading and processing documents...")
        doc_manager.load_documents()
        sections = doc_manager.split_documents()
        doc_manager.create_chroma_store(sections)
        st.success(f"Processed {len(sections)} document sections and stored in Chroma DB!")

# Section for inputting questions (basic placeholder)
st.header("Ask Questions")
question = st.text_input("Enter your question here:")

# Placeholder for output
st.header("Output")
if question:
    st.write("Your RAG-generated answer will appear here (functionality pending).")

