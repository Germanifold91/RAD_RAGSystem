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
        glob_pattern: str = "./*.md",
        embedding_model: str = "openai",
    ) -> None:
        self.directory_path = directory_path
        self.chroma_path = chroma_path
        self.glob_pattern = glob_pattern
        self.embedding_model = embedding_model
        self.documents = []
        self.all_sections = []

    def load_documents(self) -> None:
        """
        Load documents from the specified directory path using the given glob pattern.

        This method initializes a `DirectoryLoader` object with the directory path and glob pattern provided.
        It then uses the `TextLoader` class as the loader class for loading the documents.

        Returns:
            None
        """
        loader = DirectoryLoader(
            self.directory_path,
            glob=self.glob_pattern,
            show_progress=True,
            loader_cls=TextLoader,
        )
        self.documents = loader.load()

    def split_documents(self) -> List[Document]:
        """
        Splits the documents into sections based on specified headers.

        Returns a list of sections containing the split content.

        :return: A list of sections.
        :rtype: List[Document]
        """
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
        """
        Create a Chroma store and add the initial set of chunks to it.

        Args:
            chunks (List[Document]): The list of chunks to be added to the Chroma store.

        Returns:
            None
        """
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
        """
        Calculates and assigns unique IDs to each chunk in the given list of documents.

        Args:
            chunks (List[Document]): A list of Document objects representing chunks.

        Returns:
            List[Document]: A list of Document objects with assigned chunk IDs.

        """
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
        glob_pattern: str = "./*.md",
        embedding_model: str = "openai",
    ) -> None:
        super().__init__(directory_path, chroma_path, glob_pattern, embedding_model)
        self.temp_directory = tempfile.mkdtemp()

    def load_temp_documents(self) -> None:
        """
        Loads temporary documents from the specified directory using the DirectoryLoader class.

        This method initializes a DirectoryLoader with the given temporary directory, glob pattern,
        and TextLoader class. It then uses the loader to load the documents from the directory.

        Returns:
            None
        """
        loader = DirectoryLoader(
            self.temp_directory,
            glob=self.glob_pattern,
            show_progress=True,
            loader_cls=TextLoader,
        )
        self.documents = loader.load()

    def update_chroma_store(self) -> None:
        """
        Update the Chroma store with new documents.

        This method loads and splits documents from the temporary folder, calculates chunk IDs,
        and adds new documents to the Chroma store if they don't already exist. It also moves
        new documents from the temporary directory to the main directory and clears the temporary
        directory afterwards.

        Returns:
            None
        """
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
        """Move temporary files to the main directory.

        This method moves all the files from the temporary directory to the main directory.
        It iterates over each file in the temporary directory, checks if it is a file, and then
        moves it to the destination path in the main directory.

        Note:
            - The source path is the path of the file in the temporary directory.
            - The destination path is the path where the file will be moved in the main directory.

        Raises:
            - FileNotFoundError: If the temporary directory or any of the files in it does not exist.
            - PermissionError: If there is a permission error while moving the files.

        """
        for filename in os.listdir(self.temp_directory):
            source_path = os.path.join(self.temp_directory, filename)
            destination_path = os.path.join(self.directory_path, filename)
            if os.path.isfile(source_path):
                shutil.move(source_path, destination_path)
                print(
                    f"🚚 Moved {os.path.basename(source_path)} to {self.directory_path}"
                )

    def clear_temp_directory(self) -> None:
        """
        Clears the temporary directory used by the Vecstore Manager.

        This method deletes all files and subdirectories within the temporary directory
        and creates a new empty temporary directory.

        Returns:
            None
        """
        shutil.rmtree(self.temp_directory)
        self.temp_directory = tempfile.mkdtemp()

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
            print(f"File not found: {file_path}")
            return

        destination_path = os.path.join(
            self.temp_directory, os.path.basename(file_path)
        )
        shutil.copy(file_path, destination_path)
