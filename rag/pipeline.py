from .config import CHUNK_OVERLAP, CHUNK_SIZE, DOCUMENT_DIR, EMBEDDING_MODEL, INDEX_PATH, METADATA_PATH, TOP_K
from .document_loader import chunk_text, load_documents
from .embeddings import Embedder
from .index import VectorIndex
from .llm import generate_answer

class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder(EMBEDDING_MODEL)
        self.index = VectorIndex(INDEX_PATH, METADATA_PATH)

    def load_index(self):
        self.index.load()

    def retrieve(self, question, top_k=TOP_K):
        vector = self.embedder.encode([question])
        return self.index.search(vector, top_k=top_k)

    def ask(self, question, top_k=TOP_K):
        contexts = self.retrieve(question, top_k)
        return {"answer": generate_answer(question, contexts), "sources": contexts}

    def ingest(self):
        documents = load_documents(DOCUMENT_DIR)
        all_chunks, metadata = [], []
        for doc in documents:
            chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
            for number, chunk in enumerate(chunks, start=1):
                all_chunks.append(chunk)
                metadata.append({"source": doc["source"], "chunk": number, "text": chunk})
        if not all_chunks:
            raise RuntimeError("No supported documents found.")
        vectors = self.embedder.encode(all_chunks)
        self.index.build(vectors, metadata)
        return {"documents": len(documents), "chunks": len(all_chunks)}
