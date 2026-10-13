import json
from pathlib import Path
from rag.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]
cases = json.loads((ROOT / "data/evaluation/questions.json").read_text(encoding="utf-8"))
pipeline = RAGPipeline()
pipeline.load_index()

hits = 0
keyword_hits = 0
total_keywords = sum(len(c["expected_keywords"]) for c in cases)

for case in cases:
    contexts = pipeline.retrieve(case["question"], top_k=4)
    sources = [c["source"] for c in contexts]
    if case["expected_source"] in sources:
        hits += 1
    combined = " ".join(c["text"].lower() for c in contexts)
    keyword_hits += sum(k.lower() in combined for k in case["expected_keywords"])

print(f"Retrieval hit rate: {hits/len(cases):.2%}")
print(f"Context keyword coverage: {keyword_hits/total_keywords:.2%}")
