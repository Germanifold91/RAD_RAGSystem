"""
"""

import os
from typing import List
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.schema.document import Document


def calculate_chunk_ids(chunks):
    # Dictionary to keep track of the last chunk index for each source
    source_last_chunk_index = {}

    for chunk in chunks:
        source = chunk.metadata.get("source")

        if source not in source_last_chunk_index:
            # Initialize the index for a new source
            source_last_chunk_index[source] = 0
        else:
            # Increment the index for existing sources
            source_last_chunk_index[source] += 1

        # Calculate the chunk ID
        chunk_id = f"{source}:0{source_last_chunk_index[source]}"

        # Add it to the chunk metadata
        chunk.metadata["id"] = chunk_id

    return chunks


def get_embedding_function(embedding_model: str, **kwargs):
    """ """
    # Load the model
    embeddings = {
        "openai": OpenAIEmbeddings(**kwargs),
    }

    return embeddings[embedding_model]


class DocumentManager:

    """
    """
    def __init__(
        self,
        directory_path: str,
        glob_pattern: str = "./*.md",
        embedding_model: str = "openai",
    ):
        self.directory_path = directory_path
        self.glob_pattern = glob_pattern
        self.embedding_model = embedding_model
        self.documents = []
        self.all_sections = []

    def load_documents(self) -> None:
        loader = DirectoryLoader(
            self.directory_path,
            glob=self.glob_pattern,
            show_progress=True,
            loader_cls=UnstructuredMarkdownLoader,
        )
        self.documents = loader.load()

    def split_documents(self) -> List[Document]:
        """
        """
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
        text_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False
        )
        for doc in self.documents:
            sections = text_splitter.split_text(doc.page_content)
            for section in sections:
                section.metadata["source"] = os.path.basename(doc.metadata["source"])
                section.metadata["header"] = doc.page_content.split("\n")[0]
            self.all_sections.extend(sections)

        return self.all_sections

    def add_to_chroma(self, chunks: List[Document], directory_path: str) -> None:
        """
        """
        # Load the existing database.
        db = Chroma(
            persist_directory=directory_path,
            embedding_function=get_embedding_function(
                embedding_model=self.embedding_model
            ),
        )

        # Calculate chunk IDs.
        chunks_with_ids = calculate_chunk_ids(chunks)

        # Add or update the documents.
        existing_items = db.get()  # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Only add documents that don't exist in the DB.
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            print("✅ No new documents to add")
