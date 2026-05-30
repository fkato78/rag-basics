from typing import Any

from classes.chromadb_class import ChromaDBManager
from classes.openai_class import OpenAIClient


class RAGSystem:
    """A complete RAG system combining OpenAI client and ChromaDB."""

    def __init__(self, openai_client: OpenAIClient, chroma_manager: ChromaDBManager):
        """Initialize the RAG system.

        Args:
            openai_client: Instance of OpenAIClient.
            chroma_manager: Instance of ChromaDBManager.
        """
        self.openai_client = openai_client
        self.chroma_manager = chroma_manager

    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document with automatic embedding generation.

        Args:
            doc_id: Unique identifier for the document.
            text: Document text content.
        """
        embedding = self.openai_client.generate_embedding(text)
        self.chroma_manager.insert_document(doc_id, text, embedding)

    def add_documents_batch(self, documents: list[dict[str, str]]) -> int:
        """Add multiple documents with automatic embedding generation.

        Args:
            documents: List of dictionaries with 'id' and 'text' keys.

        Returns:
            Number of documents added.
        """
        chunked_documents = []

        for doc in documents:
            if "id" not in doc or "text" not in doc:
                raise ValueError("Each document must have 'id' and 'text' keys")

            embedding = self.openai_client.generate_embedding(doc["text"])
            chunked_documents.append(
                {"id": doc["id"], "text": doc["text"], "embedding": embedding}
            )

        return self.chroma_manager.insert_documents(chunked_documents)

    def query_and_answer(self, question: str, n_results: int = 3) -> dict[str, Any]:
        """Query relevant documents and generate an answer.

        Args:
            question: User's question.
            n_results: Number of context documents to retrieve.

        Returns:
            Dictionary containing the answer and retrieved context.
        """
        # Retrieve relevant documents
        context_chunks = self.chroma_manager.query_documents(question, n_results)

        # Generate answer using LLM
        answer = self.openai_client.ask_question(question, context_chunks)

        return {
            "answer": answer,
            "context_used": context_chunks,
            "num_context_chunks": len(context_chunks),
        }
