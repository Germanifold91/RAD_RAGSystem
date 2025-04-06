"""

"""

import os
import tempfile
import shutil
import logging
import glob
from collections import defaultdict
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores.chroma import Chroma
from langchain.schema.document import Document

LOGGER = logging.getLogger(__name__)


def get_embedding_function(embedding_model: str, **kwargs):
    """
    Returns the embedding function for the specified model
    Parameters:
        embedding_model (str): The name of the embedding model
        **kwargs: Additional keyword arguments to pass to the embedding model
    Returns:
        function: Embedding model function
    """
    # Load the model
    try:
        embeddings = {
            "openai": OpenAIEmbeddings(**kwargs),
        }
    except Exception as e:
        raise ValueError(f"Error loading embedding model: {e}")

    return embeddings[embedding_model]


class DocumentManager:
    """
    A class that manages loading, splitting, and storing documents.

    The `DocumentManager` class provides methods to load documents from a specified directory,
    split the documents into sections based on headers, and create a Chroma store to store the chunks of documents.

    Args:
        directory_path (str): The path to the directory containing the documents.
        chroma_path (str): The path to the directory where the Chroma store will be created.
        glob_pattern (str, optional): The glob pattern used to filter the documents. Defaults to "./*.md".
        embedding_model (str, optional): The embedding model used for creating document embeddings. Defaults to "openai".

    Attributes:
        directory_path (str): The path to the directory containing the documents.
        chroma_path (str): The path to the directory where the Chroma store will be created.
        glob_pattern (str): The glob pattern used to filter the documents.
        embedding_model (str): The embedding model used for creating document embeddings.
        documents (List[Document]): The list of loaded documents.
        all_sections (List[Document]): The list of all sections obtained after splitting the documents.

    Methods:
        load_documents: Load documents from the specified directory path using the given glob pattern.
        split_documents: Split the documents into sections based on specified headers.
        create_chroma_store: Create a Chroma store and add the initial set of chunks to it.
        calculate_chunk_ids: Calculate and assign unique IDs to each chunk in the given list of documents.
    """

    def __init__(
        self,
        directory_path: str,
        chroma_path: str,
        glob_pattern: str = "*.*",
        embedding_model: str = "openai",
    ) -> None:
        self.directory_path = directory_path
        self.chroma_path = chroma_path
        self.glob_pattern = glob_pattern
        self.embedding_model = embedding_model
        self.loader_map = {".md": TextLoader, ".pdf": PyPDFLoader}
        self.documents = []
        self.all_sections = []

    def load_documents(self) -> None:
        """
        Loads supported documents from the directory into memory.

        Supported formats:
            - Markdown (.md)
            - PDF (.pdf)

        Files with unsupported extensions are skipped with a warning.
        Any errors during document loading are logged without interrupting the process.

        Side Effects:
            Populates `self.documents` with loaded document objects.

        Raises:
            None

        Logs:
            - Number of documents loaded
            - Any skipped or failed files
        """
        all_files = glob.glob(
            os.path.join(self.directory_path, self.glob_pattern), recursive=True
        )
        docs = []

        for file_path in all_files:
            ext = os.path.splitext(file_path)[1].lower()
            loader_cls = self.loader_map.get(ext)

            if not loader_cls:
                logging.warning(f"Skipping unsupported file: {file_path}")
                continue

            try:
                loader = loader_cls(file_path)
                docs.extend(loader.load())
            except Exception as e:
                logging.error(f"Error loading {file_path}: {e}")

        self.documents = docs
        logging.info(f"Successfully loaded {len(docs)} documents.")

    def split_documents(self) -> List[Document]:
        """
        Splits loaded documents into smaller, semantically meaningful chunks.

        Markdown files (`.md`) are split using header-based segmentation with `MarkdownHeaderTextSplitter`,
        preserving header context in metadata.

        PDFs and other file formats are split using recursive character-based chunking with overlap.

        Returns:
            List[Document]: A list of chunked documents with enriched metadata.

        Raises:
            ValueError: If `self.documents` is empty or not loaded.

        Side Effects:
            Resets and repopulates `self.all_sections` with the newly created chunks.
        """
        if not self.documents:
            raise ValueError("No documents to split. Please load documents first.")

        all_sections = []  # Fresh list to avoid state contamination

        for doc in self.documents:
            filename = os.path.basename(doc.metadata.get("source", "unknown"))

            if filename.endswith(".md"):
                splitter = MarkdownHeaderTextSplitter(
                    headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")],
                    strip_headers=False,
                )
                sections = splitter.split_text(doc.page_content)

                for section in sections:
                    section.metadata["source"] = filename
                    section.metadata["header"] = doc.page_content.split("\n")[0]
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=100
                )
                sections = splitter.create_documents(
                    [doc.page_content], metadatas=[doc.metadata]
                )
                for section in sections:
                    section.metadata["source"] = filename

            all_sections.extend(sections)

        self.all_sections = all_sections  # Safe assignment at the end
        logging.info(
            f"Split {len(self.documents)} documents into {len(all_sections)} chunks."
        )
        return all_sections
    
    def calculate_chunk_ids(self, chunks: List[Document]) -> List[Document]:
        """
        Assigns unique IDs to each document chunk based on its source filename and position.

        The ID format is: `<source>:<index>`, where index is incremented per source file.

        Args:
            chunks (List[Document]): A list of Document chunks with metadata that includes 'source'.

        Returns:
            List[Document]: The same list of Document chunks with a new "id" key in metadata.

        Raises:
            ValueError: If a chunk is missing the 'source' metadata field.
        """
        source_last_chunk_index = defaultdict(int)

        for chunk in chunks:
            source = chunk.metadata.get("source")
            if not source:
                raise ValueError("Document chunk is missing 'source' metadata.")

            source_last_chunk_index[source] += 1
            chunk_id = f"{source}:{source_last_chunk_index[source]}"
            chunk.metadata["id"] = chunk_id

        return chunks

    def create_chroma_store(self, chunks: List[Document]) -> None:
        """
        Creates a Chroma vector store at the specified path and adds the provided document chunks.

        Each chunk is assigned a unique ID before being added to the store.

        Args:
            chunks (List[Document]): The list of document chunks to be embedded and stored.

        Returns:
            None

        Raises:
            FileExistsError: If a Chroma store already exists at `self.chroma_path`.
        
        Logs:
            - Number of chunks added
            - Chroma DB creation path
        """
        if os.path.exists(self.chroma_path):
            raise FileExistsError(
                f"Chroma store already exists at {self.chroma_path}. Aborting to prevent overwrite."
            )

        db = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=get_embedding_function(
                embedding_model=self.embedding_model
            ),
        )

        chunks_with_ids = self.calculate_chunk_ids(chunks)
        chunk_ids = [chunk.metadata["id"] for chunk in chunks_with_ids]
        db.add_documents(chunks_with_ids, ids=chunk_ids)

        LOGGER.info(
            f"👉 Initial set of chunks added to Chroma store: {len(chunks_with_ids)}\n📍 Chroma store created at: {self.chroma_path}"
        )


class DocumentUpdater(DocumentManager):
    """
    A class that updates the document store with new documents.

    This class extends the `DocumentManager` class and provides methods to load temporary documents,
    update the Chroma store with new documents, move temporary files to the main directory, and clear
    the temporary directory.

    Args:
        directory_path (str): The path to the main directory where the documents are stored.
        chroma_path (str): The path to the Chroma store.
        glob_pattern (str, optional): The glob pattern to match the documents in the temporary directory. Defaults to "./*.md".
        embedding_model (str, optional): The embedding model to use for calculating document embeddings. Defaults to "openai".

    Attributes:
        temp_directory (str): The path to the temporary directory.

    Methods:
        load_temp_documents(): Loads temporary documents from the specified directory.
        update_chroma_store(): Updates the Chroma store with new documents.
        move_temp_files_to_main_directory(): Moves temporary files to the main directory.
        clear_temp_directory(): Clears the temporary directory used by the Vecstore Manager.
        upload_document(file_path: str): Uploads a document from the specified file path to the temporary directory.
    """

    def __init__(
        self,
        directory_path: str,
        chroma_path: str,
        glob_pattern: str = "*.*",
        embedding_model: str = "openai",
        loader_map: dict = {".md": TextLoader, ".pdf": PyPDFLoader},
    ) -> None:
        super().__init__(directory_path, chroma_path, glob_pattern, embedding_model, loader_map)
        self.temp_directory = tempfile.mkdtemp()

    def load_temp_documents(self) -> None:
        """
        Loads supported temporary documents from the temporary directory into memory.

        This method reads files from `self.temp_directory` using glob and loads them
        using appropriate document loaders (Markdown or PDF). Unsupported file formats
        are skipped with a warning. Loaded documents are stored in `self.documents`.

        Raises:
            None

        Side Effects:
            Sets `self.documents` with the list of loaded Document objects.

        Logs:
            - Skipped unsupported file types
            - Any loading errors
            - Total number of documents loaded
        """
        all_files = glob.glob(os.path.join(self.temp_directory, self.glob_pattern))
        docs = []

        for file_path in all_files:
            ext = os.path.splitext(file_path)[1].lower()
            loader_cls = self.loader_map.get(ext)

            if not loader_cls:
                logging.warning(f"Skipping unsupported file: {file_path}")
                continue

            try:
                loader = loader_cls(file_path)
                docs.extend(loader.load())
            except Exception as e:
                logging.error(f"Failed to load {file_path}: {e}")

        self.documents = docs
        logging.info(f"Loaded {len(docs)} documents from temp directory.")

    def move_temp_files_to_main_directory(self) -> None:
        """
        Moves all files from the temporary directory to the main directory.

        For each file in the temporary directory:
            - Checks if it's a regular file (not a directory)
            - Moves it to the main directory
            - Logs the result

        Notes:
            - Ensures the destination directory exists
            - Logs success or failure for each file

        Raises:
            FileNotFoundError: If the temporary directory doesn't exist.
            PermissionError: If a file cannot be moved due to permissions.
        """
        if not os.path.exists(self.temp_directory):
            raise FileNotFoundError(f"Temporary directory not found: {self.temp_directory}")

        os.makedirs(self.directory_path, exist_ok=True)

        for entry in os.scandir(self.temp_directory):
            if entry.is_file():
                source_path = entry.path
                destination_path = os.path.join(self.directory_path, entry.name)

                try:
                    shutil.move(source_path, destination_path)
                    LOGGER.info(f"🚚 Moved {entry.name} to {self.directory_path}")
                except Exception as e:
                    LOGGER.error(f"❌ Failed to move {entry.name}: {e}")

    def clear_temp_directory(self) -> None:
        """
        Clears the temporary directory used by the Vecstore Manager.

        This method deletes all files and subdirectories within the temporary directory
        and creates a new empty one, updating `self.temp_directory`.

        Returns:
            None

        Logs:
            - Successful clearing and recreation of the directory.
        """
        if os.path.exists(self.temp_directory):
            shutil.rmtree(self.temp_directory)
            LOGGER.info(f"🗑️ Removed existing temp directory: {self.temp_directory}")

        self.temp_directory = tempfile.mkdtemp()
        LOGGER.info(f"🧹 Created new temp directory: {self.temp_directory}")

    def update_chroma_store(self) -> int:
        """
        Updates the Chroma vector store with new documents from the temporary directory.

        This method:
            - Loads and splits temporary documents
            - Assigns unique chunk IDs
            - Adds only new (non-duplicate) chunks to the Chroma store
            - Moves processed documents to the main directory
            - Clears the temporary directory

        Returns:
            int: The number of new chunks added to the Chroma store.

        Logs:
            - Number of chunks loaded, filtered, and added
            - Skipped duplicates
            - Whether new files were moved or temp folder was cleared

        Raises:
            Exception: Propagates any unrecoverable errors during Chroma access or file I/O
        """
        try:
            # Step 1: Load and split new documents
            self.load_temp_documents()
            chunks = self.split_documents()
            LOGGER.info(f"📄 Loaded and split {len(chunks)} chunks from temp documents.")

            # Step 2: Load existing Chroma DB
            db = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=get_embedding_function(
                    embedding_model=self.embedding_model
                ),
            )

            # Step 3: Assign unique chunk IDs
            chunks_with_ids = self.calculate_chunk_ids(chunks)

            # Step 4: Get existing IDs from DB
            existing_ids = set(db.get()["ids"])
            LOGGER.info(f"📦 Chroma currently holds {len(existing_ids)} vectors.")

            # Step 5: Filter out already-indexed chunks
            new_chunks = [
                chunk for chunk in chunks_with_ids
                if chunk.metadata["id"] not in existing_ids
            ]

            if new_chunks:
                chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
                db.add_documents(new_chunks, ids=chunk_ids)

                LOGGER.info(
                    f"✅ Added {len(new_chunks)} new chunks to Chroma store."
                )

                # Step 6: Move source files to main directory
                self.move_temp_files_to_main_directory()
                LOGGER.info("📂 Moved processed documents to main directory.")

            else:
                LOGGER.info("👍 No new chunks to add. All documents were already indexed.")

            # Step 7: Clean up
            self.clear_temp_directory()
            LOGGER.info("🧹 Temporary directory cleared.")

            return len(new_chunks)

        except Exception as e:
            LOGGER.error(f"❌ Failed to update Chroma store: {e}")
            raise

    def upload_document(self, file_path: str) -> None:
        """
        Uploads a document from the specified file path to the temporary directory.

        Args:
            file_path (str): The path of the file to be uploaded.

        Returns:
            None

        Raises:
            None

        """
        if not os.path.isfile(file_path):
            LOGGER.info(f"File not found: {file_path}")
            return

        destination_path = os.path.join(
            self.temp_directory, os.path.basename(file_path)
        )
        shutil.copy(file_path, destination_path)
