"""
"""

import os
import tempfile
import shutil
from typing import List
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.schema.document import Document



def get_embedding_function(embedding_model: str, **kwargs):
    """ """
    # Load the model
    embeddings = {
        "openai": OpenAIEmbeddings(**kwargs),
    }

    return embeddings[embedding_model]


class DocumentManager:
    """ """

    def __init__(
        self,
        directory_path: str,
        chroma_path: str,
        glob_pattern: str = "./*.md",
        embedding_model: str = "openai",
    ) -> None:
        """ """
        self.directory_path = directory_path
        self.chroma_path = chroma_path
        self.glob_pattern = glob_pattern
        self.embedding_model = embedding_model
        self.documents = []
        self.all_sections = []

    def load_documents(self) -> None:
        """ """
        loader = DirectoryLoader(
            self.directory_path,
            glob=self.glob_pattern,
            show_progress=True,
            loader_cls=TextLoader,
        )
        self.documents = loader.load()

    def split_documents(self) -> List[Document]:
        """ """
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
        self.text_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False
        )
        for doc in self.documents:
            sections = self.text_splitter.split_text(doc.page_content)
            for section in sections:
                section.metadata["source"] = os.path.basename(doc.metadata["source"])
                section.metadata["header"] = doc.page_content.split("\n")[0]
            self.all_sections.extend(sections)

        return self.all_sections

    def create_chroma_store(self, chunks: List[Document]) -> None:
        """ """
        db = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=get_embedding_function(
                embedding_model=self.embedding_model
            ),
        )

        chunks_with_ids = self.calculate_chunk_ids(chunks)
        chunk_ids = [chunk.metadata["id"] for chunk in chunks_with_ids]
        db.add_documents(chunks, ids=chunk_ids)

        print(
            f"👉 Initial set of chunks added to chroma store: {len(chunks_with_ids)}\n📍 Chroma store created at: {self.chroma_path}"
        )

    def calculate_chunk_ids(self, chunks: List[Document]) -> List[Document]:
        """ """
        source_last_chunk_index = {}
        for chunk in chunks:
            source = chunk.metadata.get("source")
            if source not in source_last_chunk_index:
                source_last_chunk_index[source] = 0
            else:
                source_last_chunk_index[source] += 1
            chunk_id = f"{source}:{source_last_chunk_index[source]}"
            chunk.metadata["id"] = chunk_id
        return chunks


class DocumentUpdater(DocumentManager):
    """ """

    def __init__(
        self,
        directory_path: str,
        chroma_path: str,
        glob_pattern: str = "./*.md",
        embedding_model: str = "openai",
    ) -> None:
        super().__init__(directory_path, chroma_path, glob_pattern, embedding_model)
        self.temp_directory = tempfile.mkdtemp()

    def load_temp_documents(self) -> None:
        """ """
        loader = DirectoryLoader(
            self.temp_directory,
            glob=self.glob_pattern,
            show_progress=True,
            loader_cls=TextLoader,
        )
        self.documents = loader.load()

    def update_chroma_store(self) -> None:
        """ """
        # Load and split documents from the temporary folder
        self.load_temp_documents()
        chunks = self.split_documents()

        # Load the existing database
        db = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=get_embedding_function(
                embedding_model=self.embedding_model
            ),
        )

        # Calculate chunk IDs
        chunks_with_ids = self.calculate_chunk_ids(chunks)

        # Get existing IDs from the database
        existing_items = db.get()  # IDs are always included by default
        existing_ids = set(existing_items["ids"])

        # Only add documents that don't exist in the DB
        new_chunks = [
            chunk
            for chunk in chunks_with_ids
            if chunk.metadata["id"] not in existing_ids
        ]

        if new_chunks:
            chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=chunk_ids)
            print(f"👉 Added {len(new_chunks)} new vectors to the Chroma store.")

            # Move new documents from temp directory to main directory
            self.move_temp_files_to_main_directory()

        else:
            print("✅ No new documents to add.")

        # Clear the temporary directory
        self.clear_temp_directory()

    def move_temp_files_to_main_directory(self) -> None:
        """ """
        for filename in os.listdir(self.temp_directory):
            source_path = os.path.join(self.temp_directory, filename)
            destination_path = os.path.join(self.directory_path, filename)
            if os.path.isfile(source_path):
                shutil.move(source_path, destination_path)
                print(f"🚚 Moved {os.path.basename(source_path)} to {self.directory_path}")

    def clear_temp_directory(self) -> None:
        """ """
        shutil.rmtree(self.temp_directory)
        self.temp_directory = tempfile.mkdtemp()

    def upload_document(self, file_path: str) -> None:
        """Uploads a document to the temporary directory for processing"""
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            return

        destination_path = os.path.join(
            self.temp_directory, os.path.basename(file_path)
        )
        shutil.copy(file_path, destination_path)