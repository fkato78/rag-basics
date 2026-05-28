# import os

# import chromadb
# from chromadb.utils import embedding_functions
# from dotenv import load_dotenv

# load_dotenv()

# api_key = os.getenv("API_KEY")


# openai_ef = embedding_functions.OpenAIEmbeddingFunction(
#     api_key=api_key,
#     model_name="text-embedding-3-small",
# )

# chroma_client = chromadb.PersistentClient(path="./db/chroma_db")
# collection_name = "documents_qa_collection"

# collection = chroma_client.get_or_create_collection(
#     name=collection_name, embedding_function=openai_ef
# )
