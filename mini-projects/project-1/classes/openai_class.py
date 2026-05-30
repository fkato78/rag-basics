import os

from dotenv import load_dotenv
from openai import OpenAI


class OpenAIClient:
    """A wrapper class for OpenAI API interactions."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        """Initialize the OpenAI client with API key and model.

        Args:
            api_key: OpenAI API key. If None, loads from environment.
            model: AI model to use. If None, loads from environment or defaults to "gpt-5-nano".
        """
        load_dotenv()

        self.api_key = api_key or os.getenv("API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. Set API_KEY in environment or pass it directly."
            )

        self.ai_model = model or os.getenv("AI_MODEL") or "gpt-5-nano"
        self.embedding_model = "text-embedding-3-small"

        self.client = OpenAI(api_key=self.api_key)

    def generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Args:
            text: The input text to encode.

        Returns:
            The embedding vector returned by the OpenAI API.

        Raises:
            ValueError: If the response doesn't contain an embedding.
        """
        response = self.client.embeddings.create(input=text, model=self.embedding_model)

        if not response.data or not response.data[0].embedding:
            raise ValueError("Failed to generate embedding")

        return response.data[0].embedding

    def ask_question(self, question: str, context_chunks: list[str]) -> str:
        """Ask a question to the LLM with provided context.

        Args:
            question: The user's question.
            context_chunks: List of text chunks to use as context.

        Returns:
            The LLM's response.

        Raises:
            ValueError: If the response content is None.
        """
        context = "\n\n---\n\n".join(context_chunks)

        response = self.client.chat.completions.create(
            model=self.ai_model,
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
