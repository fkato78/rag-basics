import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("API_KEY")
ai_model = os.getenv("AI_MODEL") or "gpt-5-nano"

openAIClient = OpenAI(api_key=api_key)


# Function to generate embeddings using OpenAI API
def generate_openai_embedding(text: str):
    """Generate an embedding vector for the given text using OpenAI.

    Args:
        text (str): The input text to encode.

    Returns:
        list[float]: The embedding vector returned by the OpenAI API.
    """
    response = openAIClient.embeddings.create(
        input=text, model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    return embedding


def ask_llm(question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)

    response = openAIClient.chat.completions.create(
        model=ai_model,
        temperature=1,
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
