import os
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

# from openai_client import (
#     OpenAIClient,  # Assuming the previous class is in openai_client.py
# )


class ChromaDBManager:
    """A wrapper class for ChromaDB operations."""

    def __init__(
        self,
        collection_name: str = "project-1",
        persist_directory: str = "./db/chroma_db",
        api_key: str | None = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        """Initialize the ChromaDB manager.

        Args:
            collection_name: Name of the collection to use.
            persist_directory: Directory path for persistent storage.
            api_key: OpenAI API key. If None, loads from environment.
            embedding_model: Embedding model to use.
        """
        load_dotenv()

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.api_key = api_key or os.getenv("API_KEY")

        if not self.api_key:
            raise ValueError(
                "API key is required. Set API_KEY in environment or pass it directly."
            )

        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)

        # Get or create collection with embedding function
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=OpenAIEmbeddingFunction(
                api_key=self.api_key, model_name=embedding_model
            ),
        )

        # Store embedding model name for reference
        self.embedding_model = embedding_model

    def insert_documents(self, chunked_documents: list[dict[str, Any]]) -> int:
        """Insert or update documents in ChromaDB.

        Args:
            chunked_documents: List of dictionaries containing 'id', 'text', and 'embedding' keys.

        Returns:
            Number of documents inserted.

        Raises:
            ValueError: If documents are missing required keys.
        """
        if not chunked_documents:
            return 0

        # Validate documents
        for doc in chunked_documents:
            if not all(key in doc for key in ["id", "text", "embedding"]):
                raise ValueError(
                    "Each document must have 'id', 'text', and 'embedding' keys"
                )

        # Batch upsert for better performance
        ids = [doc["id"] for doc in chunked_documents]
        texts = [doc["text"] for doc in chunked_documents]
        embeddings = [doc["embedding"] for doc in chunked_documents]

        self.collection.upsert(ids=ids, documents=texts, embeddings=embeddings)

        return len(chunked_documents)

    def insert_document(self, doc_id: str, text: str, embedding: list[float]) -> None:
        """Insert or update a single document in ChromaDB.

        Args:
            doc_id: Unique identifier for the document.
            text: Document text content.
            embedding: Pre-computed embedding vector.
        """
        self.collection.upsert(ids=[doc_id], documents=[text], embeddings=[embedding])

    def query_documents(
        self, question: str, n_results: int = 3, include_embeddings: bool = False
    ) -> list[str]:
        """Query documents similar to the question.

        Args:
            question: The query text.
            n_results: Number of results to return.
            include_embeddings: Whether to include embeddings in results.

        Returns:
            List of document texts.
        """
        include_fields = ["documents", "distances"]
        if include_embeddings:
            include_fields.append("embeddings")

        results = self.collection.query(
            query_texts=question, n_results=n_results, include=include_fields
        )

        print(
            f"Query results: {results}"
        )  # Debugging statement to check the structure of results

        # Return the first list of documents (from the first query)
        return results["documents"][0] if results["documents"] else []

    def query_documents_with_metadata(
        self, question: str, n_results: int = 3
    ) -> dict[str, Any]:
        """Query documents and return full results with metadata.

        Args:
            question: The query text.
            n_results: Number of results to return.

        Returns:
            Dictionary containing ids, documents, distances, and metadata.
        """
        results = self.collection.query(
            query_texts=question,
            n_results=n_results,
            include=["documents", "distances", "metadatas", "embeddings"],
        )

        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "embeddings": results["embeddings"][0] if results["embeddings"] else [],
        }

    def delete_documents(self, doc_ids: list[str]) -> int:
        """Delete documents from the collection.

        Args:
            doc_ids: List of document IDs to delete.

        Returns:
            Number of documents deleted.
        """
        if not doc_ids:
            return 0

        self.collection.delete(ids=doc_ids)
        return len(doc_ids)

    def get_document_count(self) -> int:
        """Get the total number of documents in the collection.

        Returns:
            Number of documents in the collection.
        """
        return self.collection.count()

    def clear_collection(self) -> None:
        """Delete all documents in the collection."""
        all_ids = self.collection.get()["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)

    def list_collections(self) -> list[str]:
        """List all available collections.

        Returns:
            List of collection names.
        """
        collections = self.chroma_client.list_collections()
        return [col.name for col in collections]
