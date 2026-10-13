import logging

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENT_DIR,
    EMBEDDING_MODEL,
    INDEX_PATH,
    METADATA_PATH,
    TOP_K,
)
from .document_loader import chunk_text, load_documents
from .embeddings import Embedder
from .index import VectorIndex
from .llm import generate_answer

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self) -> None:
        self.embedder = Embedder(EMBEDDING_MODEL)
        self.index = VectorIndex(INDEX_PATH, METADATA_PATH)

    def load_index(self) -> None:
        self.index.load()

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[dict]:
        vector = self.embedder.encode([question])
        return self.index.search(vector, top_k=top_k)

    def ask(self, question: str, top_k: int = TOP_K) -> dict:
        contexts = self.retrieve(question, top_k)
        answer = generate_answer(question, contexts)
        return {"answer": answer, "sources": contexts}

    def ingest(self) -> dict:
        logger.info("Loading documents from %s", DOCUMENT_DIR)
        documents = load_documents(DOCUMENT_DIR)
        if not documents:
            raise RuntimeError(
                f"No supported documents found in {DOCUMENT_DIR}. "
                "Add .txt, .md, or .pdf files and rerun."
            )

        all_chunks: list[str] = []
        metadata: list[dict] = []

        for doc in documents:
            chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
            for number, chunk in enumerate(chunks, start=1):
                all_chunks.append(chunk)
                metadata.append(
                    {"source": doc["source"], "chunk": number, "text": chunk}
                )
            logger.info("  %s → %d chunks", doc["source"], len(chunks))

        logger.info("Encoding %d chunks…", len(all_chunks))
        vectors = self.embedder.encode(all_chunks)
        self.index.build(vectors, metadata)
        logger.info("Index built and saved.")
        return {"documents": len(documents), "chunks": len(all_chunks)}
