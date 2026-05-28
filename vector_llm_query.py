# from chroma_client import collection


# # Function to query documents
# def query_documents(question: str):
#     # query_embedding = get_openai_embedding(question)
#     results = collection.query(query_texts=question)

#     # Extract the relevant chunks
#     relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
#     print("==== Returning relevant chunks ====")
#     return relevant_chunks
#     # for idx, document in enumerate(results["documents"][0]):
#     #     doc_id = results["ids"][0][idx]
#     #     distance = results["distances"][0][idx]
#     #     print(f"Found document chunk: {document} (ID: {doc_id}, Distance: {distance})")


# question = "Tell me about AI replacing tv writer strikes"
# relevant_chunks = query_documents(question)
# print(relevant_chunks)
