import os

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")
chroma_client = chromadb.PersistentClient(path="./db/chroma_db")


collection = chroma_client.get_or_create_collection(
    name="project-1",
    embedding_function=OpenAIEmbeddingFunction(
        api_key=api_key, model_name="text-embedding-3-small"
    ),
)


# Upsert documents with embeddings into Chroma
def insert_into_chroma(collection: chromadb.Collection, chunked_documents: list[dict]):
    for doc in chunked_documents:
        collection.upsert(
            ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]]
        )


# Function to query documents
def query_documents(question: str, n_results: int = 5) -> list[str]:
    results = collection.query(query_texts=question, n_results=n_results)

    return results["documents"][0]
