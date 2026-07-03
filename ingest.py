"""
ingest.py
---------
CLI utility to (re)ingest all PDFs in data/course_pdfs into ChromaDB.

Usage:
    python backend/ingest.py
"""

from rag_engine import RAGEngine, PDF_DIR


def main():
    engine = RAGEngine()
    print(f"Ingesting PDFs from: {PDF_DIR}")
    n_chunks = engine.ingest_directory(PDF_DIR)
    print(f"Done. {n_chunks} chunks embedded and persisted to ChromaDB.")


if __name__ == "__main__":
    main()
