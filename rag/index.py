import json
from pathlib import Path
import faiss
import numpy as np

class VectorIndex:
    def __init__(self, index_path: Path, metadata_path: Path):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []

    def build(self, vectors, metadata):
        vectors = np.asarray(vectors, dtype="float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self.index = index
        self.metadata = metadata
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        self.metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self):
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("Vector index not found. Run: python scripts/ingest.py")
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def search(self, query_vector, top_k=4):
        query_vector = np.asarray(query_vector, dtype="float32")
        scores, ids = self.index.search(query_vector, top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results
