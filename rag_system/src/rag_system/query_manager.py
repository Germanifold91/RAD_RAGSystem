"""
Enhanced query_manager.py with improvements including automatic setup, clearer naming,
basic error handling, and additional debugging and runtime methods.
"""

from langchain_openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
import logging


class ConversationalRetrievalAgent:
    def __init__(self, vectordb, temperature=0.5):
        self.vectordb = vectordb
        self.llm = OpenAI(temperature=temperature)
        self.chat_history = []
        self.setup_conversational_chain()

    def format_chat_history(self, history):
        formatted_history = []
        for human, ai in history:
            formatted_history.append(f"Human: {human}\nAI: {ai}")
        return "\n".join(formatted_history)

    def setup_conversational_chain(self):
        try:
            retriever = self.vectordb.as_retriever(search_kwargs={"k": 2})
            self.chain = ConversationalRetrievalChain.from_llm(
                self.llm,
                retriever,
                return_source_documents=True,
                get_chat_history=self.format_chat_history,
            )
        except Exception as e:
            logging.error(f"Failed to initialize conversational chain: {e}")
            raise

    def generate_prompt(self, question):
        context_instruction = (
            "You are an AI assistant in charge of helping users answer their questions from a given context. "
            "Whenever possible provide the answers as bullet points as well as code snippets found in the context that are relevant to the question. "
            "Please answer the question given the context below:"
        )

        if not self.chat_history:
            prompt = (
                f"{context_instruction}\n" f"Question: {question}\nContext: \nAnswer:"
            )
        else:
            recent_context = "\n\n".join(
                [f"Question: {q}\nAnswer: {a}" for q, a in self.chat_history[-3:]]
            )
            prompt = (
                f"{context_instruction}\n\n"
                "Using the context provided by recent conversations, answer the new question in a concise and "
                "informative way. Limit your answer to a maximum of three sentences.\n\n"
                f"Context of recent conversations:\n{recent_context}\n\n"
                f"New question: {question}\nAnswer:"
            )
        return prompt

    def ask_question(self, query):
        prompt = self.generate_prompt(query)
        try:
            result = self.chain.invoke(
                {"question": prompt, "chat_history": self.chat_history}
            )
            answer = result.get("answer", "I'm sorry, I couldn't find an answer.")
            self.chat_history.append((query, answer))
            return answer
        except Exception as e:
            logging.error(f"Failed to generate response: {e}")
            return "An error occurred while generating the response. Please try again."

    def get_source_documents(self, query):
        """Retrieves source documents using the retriever with updated method."""
        try:
            retriever = self.vectordb.as_retriever(search_kwargs={"k": 2})
            docs = retriever.invoke(query)  
            return docs
        except Exception as e:
            logging.error(f"Failed to retrieve source documents: {e}")
            return []

    def set_temperature(self, temperature):
        """Adjusts the LLM's temperature at runtime."""
        self.llm.temperature = temperature
