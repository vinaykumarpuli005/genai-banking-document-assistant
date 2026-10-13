"""
Build (or rebuild) the FAISS vector index from documents in data/documents/.

Usage:
    python -m scripts.ingest
"""
import logging

from rag.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


if __name__ == "__main__":
    result = RAGPipeline().ingest()
    print(f"\n✅ Ingested {result['documents']} document(s) → {result['chunks']} chunks.")
