"""Build the local Chroma knowledge base from the PDFs in data/.

search_knowledge_base() in tools.py only reads an existing Chroma store —
nothing in the project previously populated it. Run this once, and again
whenever files in data/ change:

    python -m src.ingest
"""
import os

import pdfplumber
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += step
    return chunks


def load_pdf_chunks(path: str) -> list[dict]:
    chunks = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            for chunk in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
                if chunk.strip():
                    chunks.append({
                        "text": chunk,
                        "metadata": {"source": os.path.basename(path), "page": page_number},
                    })
    return chunks


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"No data directory found at {DATA_DIR}")
        return

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs found in {DATA_DIR}")
        return

    texts, metadatas = [], []
    for filename in pdf_files:
        path = os.path.join(DATA_DIR, filename)
        print(f"Processing {filename}...")
        for chunk in load_pdf_chunks(path):
            texts.append(chunk["text"])
            metadatas.append(chunk["metadata"])

    print(f"Embedding {len(texts)} chunks from {len(pdf_files)} PDFs...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    db.add_texts(texts=texts, metadatas=metadatas)
    print(f"Done. Knowledge base persisted to {CHROMA_DIR}")


if __name__ == "__main__":
    main()
