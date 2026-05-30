import os

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")

if not api_key:
    raise ValueError("API_KEY is missing")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key,
    model_name="text-embedding-3-small",
)

chroma_client = chromadb.PersistentClient(path="../project-1/db/chroma_db")

collection = chroma_client.get_or_create_collection(
    name="my_project",
    embedding_function=openai_ef,
)


# Simple evaluation dataset
test_cases = [
    {
        "question": "What is Redis?",
        "expected_source": "redis.txt",
    },
    {
        "question": "What is Redis Sentinel?",
        "expected_source": "redis.txt",
    },
    {
        "question": "How does Redis support horizontal scaling?",
        "expected_source": "redis.txt",
    },
]


def retrieve(question: str, n_results: int = 3) -> dict:
    return collection.query(
        query_texts=[question],
        n_results=n_results,
        include=["documents", "distances", "metadatas"],
    )


def evaluate_retrieval() -> None:
    total = len(test_cases)
    passed = 0

    for case in test_cases:
        question = case["question"]
        expected_source = case["expected_source"]

        results = retrieve(question, n_results=3)

        ids = results["ids"][0]
        documents = results["documents"][0]
        distances = results["distances"][0]

        # Simple rule:
        # If any retrieved id starts with expected source, retrieval passes.
        is_pass = any(doc_id.startswith(expected_source) for doc_id in ids)

        if is_pass:
            passed += 1

        print("=" * 80)
        print(f"Question: {question}")
        print(f"Expected source: {expected_source}")
        print(f"Retrieved IDs: {ids}")
        print(f"Result: {'PASS ✅' if is_pass else 'FAIL ❌'}")
        print()

        for i, (doc_id, doc, distance) in enumerate(
            zip(ids, documents, distances),
            start=1,
        ):
            print(f"Rank {i}")
            print(f"ID: {doc_id}")
            print(f"Distance: {distance:.4f}")
            print(f"Preview: {doc[:250]}")
            print("-" * 80)

    accuracy = passed / total

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}/{total}")
    print(f"Retrieval Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    evaluate_retrieval()
