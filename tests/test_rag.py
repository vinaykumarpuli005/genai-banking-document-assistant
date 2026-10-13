from rag.document_loader import chunk_text
from rag.prompts import build_user_prompt

def test_chunking_returns_multiple_chunks():
    text = " ".join(["banking"] * 2000)
    assert len(chunk_text(text, 100, 20)) > 1

def test_prompt_contains_question_and_source():
    prompt = build_user_prompt("What is the limit?", [
        {"source":"policy.md","chunk":1,"text":"Limit is INR 50,000."}
    ])
    assert "What is the limit?" in prompt
    assert "policy.md" in prompt
    assert "chunk 1" in prompt
