"""
Enhanced query_manager.py with improvements including automatic setup, clearer naming,
basic error handling, and additional debugging and runtime methods.
"""

from langchain_openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from typing import List, Optional
import logging

LOGGER = logging.getLogger(__name__)


class ConversationalRetrievalAgent:
    """
    A conversational agent that integrates a language model with a document retriever
    to answer user queries based on retrieved context.

    This class supports:
        - Retrieving relevant document chunks from a Chroma vectorstore
        - Formatting prompts using both retrieved context and recent chat history
        - Generating responses using a language model (LLM)
        - Tracking conversation history for multi-turn interactions
        - Displaying source documents associated with each answer

    Attributes:
        vectordb (Chroma): The vectorstore used to retrieve relevant documents.
        llm (OpenAI): The language model used to generate responses.
        chat_history (List[Tuple[str, str]]): The history of user questions and model answers.
        k (int): The number of top-k documents to retrieve.
        retriever: A retriever instance derived from the vectorstore.
    """
    def __init__(
        self,
        vectordb: Chroma,
        temperature: float = 0.5,
        k: int = 2
    ) -> None:
        """
        Initializes the conversational retrieval agent.

        Args:
            vectordb (Chroma): The vectorstore used to retrieve context documents.
            temperature (float, optional): Temperature setting for the LLM. Defaults to 0.5.
            k (int, optional): Number of top documents to retrieve. Defaults to 2.
        """
        self.vectordb = vectordb
        self.llm = OpenAI(temperature=temperature)
        self.chat_history = []
        self.k = k
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": self.k})
        logging.info("✅ Custom retriever initialized.")

    def generate_prompt(self, question: str, context_chunks: Optional[str] = "") -> str:
        """
        Generates a structured prompt for the LLM using retrieved context and chat history.

        Args:
            question (str): The new user question.
            context_chunks (str, optional): The retrieved context from the vectorstore.

        Returns:
            str: A formatted prompt ready for the LLM.
        """
        context_instruction = (
            "You are an AI assistant in charge of helping users answer their questions from a given context. "
            "Whenever possible, provide the answers as bullet points and include any relevant code snippets found in the context. "
            "Please answer the question given the context below:"
        )

        if not self.chat_history:
            prompt = (
                f"{context_instruction}\n\n"
                f"Context:\n{context_chunks}\n\n"
                f"Question: {question}\nAnswer:"
            )
        else:
            recent_context = "\n\n".join(
                [f"Question: {q}\nAnswer: {a}" for q, a in self.chat_history[-3:]]
            )
            prompt = (
                f"{context_instruction}\n\n"
                "Using the context provided by recent conversations and retrieved documents, "
                "answer the new question concisely. Use bullet points and include code if helpful.\n\n"
                f"Recent conversation history:\n{recent_context}\n\n"
                f"Retrieved context:\n{context_chunks}\n\n"
                f"New question: {question}\nAnswer:"
            )

        return prompt

    def ask_question(self, query: str) -> str:
        """
        Handles a user query by retrieving relevant documents, generating a custom prompt,
        invoking the LLM, and formatting the final response with sources.

        Args:
            query (str): The user question to be answered.

        Returns:
            str: The LLM-generated answer, followed by a list of source documents.
        """
        try:
            # Retrieve documents manually
            retrieved_docs = self.retriever.get_relevant_documents(query)
            context_chunks = "\n\n".join([doc.page_content for doc in retrieved_docs])

            if not context_chunks.strip():
                return "🤔 I searched the documents but couldn't find anything related. Try rephrasing your question?"

            # Generate custom prompt
            prompt = self.generate_prompt(query, context_chunks)

            # Get answer from LLM
            answer = self.llm.invoke(prompt)
            self.chat_history.append((query, answer))

            # Append source filenames for traceability
            sources = "\n".join([f"\u2022 {doc.metadata.get('source', 'unknown')}" for doc in retrieved_docs])
            return f"{answer}\n\n🔎 Sources:\n{sources}"

        except Exception as e:
            logging.error(f"❌ Failed to generate response: {e}")
            return "An error occurred while generating the response. Please try again."

    def get_source_documents(self, query: str) -> List[Document]:
        """
        Retrieves relevant documents from the vectorstore for the given query.

        Args:
            query (str): The user query to retrieve relevant documents for.

        Returns:
            List[Document]: A list of retrieved document chunks.
        """
        try:
            return self.retriever.get_relevant_documents(query)
        except Exception as e:
            logging.error(f"Failed to retrieve source documents: {e}")
            return []

    def set_temperature(self, temperature: float):
        """
        Adjusts the temperature setting for the LLM.

        Args:
            temperature (float): The new temperature value to use for the LLM.
        """
        self.llm.temperature = temperature
