from classes.chromadb_class import ChromaDBManager
from classes.openai_class import OpenAIClient
from classes.rag_class import RAGSystem
from utils import load_documents_from_directory, split_text_by_paragraphs

# init classes
openai_client = OpenAIClient()
chroma_manager = ChromaDBManager(collection_name="my_project")
rag_system = RAGSystem(openai_client=openai_client, chroma_manager=chroma_manager)


def insert():
    # 1- Load all documents
    docs = load_documents_from_directory("../../data/new_articles")
    print(f"Loaded {len(docs)} documents.")

    # 2- Create chunks to be converted to embeddings
    chunked_documents = []
    for doc in docs:
        chunks = split_text_by_paragraphs(text=doc["text"])
        for i, chunk in enumerate(chunks):
            chunked_documents.append({"id": f"{doc['id']}_chunk{i + 1}", "text": chunk})
    print("==== STARTED GENERATING EMBEDDINGS AND INSERTING INTO DB ====")
    # Use RAG system to add documents with automatic embedding generation
    n = rag_system.add_documents_batch(chunked_documents)
    print(f"Inserted {n} documents into ChromaDB.")


def query():
    # Query and get answer
    response = rag_system.query_and_answer("What is Redis?")
    print(f"Answer: {response['answer']}")
    print(f"Context used: {response['context_used']}")


if __name__ == "__main__":
    # insert()
    query()
