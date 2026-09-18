"""One-time script: embed the PDFs in data/ into the local Chroma store.

`search_knowledge_base` (src/tools.py) only reads from the persisted Chroma
directory — nothing else in this repo ever populates it. Run this script
once (and again whenever data/ changes) before using that tool.
"""
import os

import pdfplumber
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))


def load_pdf_text(path: str) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    step = max(size - overlap, 1)
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


def main():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    texts, metadatas = [], []
    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.lower().endswith(".pdf"):
            continue
        text = load_pdf_text(os.path.join(DATA_DIR, filename))
        for chunk in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            texts.append(chunk)
            metadatas.append({"source": filename})

    if not texts:
        print(f"No PDFs found in {DATA_DIR}")
        return

    db.add_texts(texts, metadatas=metadatas)
    print(f"Embedded {len(texts)} chunks from {len(os.listdir(DATA_DIR))} file(s) into {CHROMA_DIR}")


if __name__ == "__main__":
    main()
