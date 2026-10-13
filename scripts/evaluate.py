"""
Evaluate retrieval quality against labelled questions.

Usage:
    python -m scripts.evaluate
"""
import json
import logging
from pathlib import Path

from rag.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "data" / "evaluation" / "questions.json"


def main() -> None:
    cases = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    pipeline = RAGPipeline()
    pipeline.load_index()

    hits = 0
    keyword_hits = 0
    total_keywords = sum(len(c["expected_keywords"]) for c in cases)

    print(f"\n{'Q#':<4} {'Hit':>4}  Question")
    print("-" * 60)

    for i, case in enumerate(cases, start=1):
        contexts = pipeline.retrieve(case["question"], top_k=4)
        sources = [c["source"] for c in contexts]
        hit = case["expected_source"] in sources
        if hit:
            hits += 1

        combined = " ".join(c["text"].lower() for c in contexts)
        kw_found = sum(k.lower() in combined for k in case["expected_keywords"])
        keyword_hits += kw_found

        status = "✅" if hit else "❌"
        print(f"{i:<4} {status}     {case['question'][:55]}")

    print("-" * 60)
    print(f"\nRetrieval hit rate       : {hits / len(cases):.2%}  ({hits}/{len(cases)})")
    print(f"Context keyword coverage : {keyword_hits / total_keywords:.2%}  ({keyword_hits}/{total_keywords})\n")


if __name__ == "__main__":
    main()
