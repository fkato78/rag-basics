import os
import re


def load_documents_from_directory(directory_path: str) -> list[dict]:
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents


def split_text_into_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[str]:

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        chunks.append(text[start:end])

        start = end - overlap
    return chunks


def split_text_by_paragraphs(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
            current_chunk += ("\n\n" if current_chunk else "") + paragraph
        else:
            if current_chunk:
                chunks.append(current_chunk)

            overlap_text = current_chunk[-overlap:] if current_chunk else ""
            current_chunk = overlap_text + "\n\n" + paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def split_text_by_sentences(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)

            overlap_text = current_chunk[-overlap:] if current_chunk else ""
            current_chunk = overlap_text + " " + sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
