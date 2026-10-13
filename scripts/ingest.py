from rag.pipeline import RAGPipeline

if __name__ == "__main__":
    result = RAGPipeline().ingest()
    print(f"Ingested {result['documents']} documents into {result['chunks']} chunks.")
