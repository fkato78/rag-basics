import os

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("AI_BASE_URL")
ai_model = os.getenv("AI_MODEL") or "gpt-5-nano"

client = OpenAI(
    api_key=api_key,
    # base_url=base_url,
)


openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key,
    model_name="text-embedding-3-small",
)

chroma_client = chromadb.PersistentClient(path="./db/chroma_db")
collection_name = "documents_qa_collection"

collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=openai_ef
)

## Load documents, split into chunks, generate embeddings, and insert into Chroma
# Uncomment the following code to run the data ingestion process. Make sure to have your .env file set up with the correct API key and base URL for the OpenAI API.
# def load_documents_from_directory(directory_path: str) -> list[dict]:
#     print("==== Loading documents from directory ====")
#     documents = []
#     for filename in os.listdir(directory_path):
#         if filename.endswith(".txt"):
#             with open(
#                 os.path.join(directory_path, filename), "r", encoding="utf-8"
#             ) as file:
#                 documents.append({"id": filename, "text": file.read()})
#     return documents


# def split_text_into_chunks(
#     text: str,
#     chunk_size: int = 1000,
#     overlap: int = 200,
# ) -> list[str]:

#     if overlap >= chunk_size:
#         raise ValueError("overlap must be smaller than chunk_size")

#     chunks: list[str] = []

#     start = 0

#     while start < len(text):
#         end = start + chunk_size

#         chunks.append(text[start:end])

#         start = end - overlap

#     return chunks


# # Load all documents
# docs = load_documents_from_directory("./data/new_articles")

# # Create chunks to be inserted in vector DB
# chunked_documents = []
# for doc in docs:
#     chunks = split_text_into_chunks(text=doc["text"])
#     for i, chunk in enumerate(chunks):
#         chunked_documents.append({"id": f"{doc['id']}_chunk{i + 1}", "text": chunk})


# # Function to generate embeddings using OpenAI API
def get_openai_embedding(text: str):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    return embedding


# # data = doc["embedding"] = get_openai_embedding(chunked_documents[0]["text"])
# # print(data)

# # Generate embeddings for the document chunks
# for doc in chunked_documents:
#     print("==== Generating embeddings... ====")
#     doc["embedding"] = get_openai_embedding(doc["text"])


# # Upsert documents with embeddings into Chroma
# for doc in chunked_documents:
#     print("==== Inserting chunks into db;;; ====")
#     collection.upsert(
#         ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]]
#     )


###
# ENDS of insert: uncomment to insert
####
# Function to query documents
def query_documents(n_results: int = 5) -> list[str]:
    # query_embedding = get_openai_embedding(question)
    results = collection.query(query_texts=question, n_results=n_results)

    return results["documents"][0]


def ask_llm(question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)

    response = client.chat.completions.create(
        model=ai_model,
        # temperature=0,
        messages=[
            {
                "role": "system",
                "content": """
You are a helpful assistant.

Answer the user's question using only the provided context.
If the answer is not in the context, say you don't know.
""",
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{question}
""",
            },
        ],
    )

    answer = response.choices[0].message.content

    if answer is None:
        raise ValueError("AI response content was None")

    return answer.strip()


question = "Tell me about Hugging Face"
relevant_chunks = query_documents()

answer = ask_llm(question, relevant_chunks)
print("==== Question ====")
print(question)
print("\n==== Answer ====")
print(answer)
