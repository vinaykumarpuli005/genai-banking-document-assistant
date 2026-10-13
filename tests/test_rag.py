"""
Unit tests for the RAG pipeline components.
Run with: pytest -v
"""
import pytest

from rag.document_loader import chunk_text, load_documents
from rag.prompts import SYSTEM_PROMPT, build_user_prompt

# ── chunk_text ────────────────────────────────────────────────────────────────

def test_chunk_text_produces_multiple_chunks():
    text = " ".join(["word"] * 2000)
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1


def test_chunk_text_respects_chunk_size():
    text = " ".join(["word"] * 500)
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    for chunk in chunks:
        assert len(chunk.split()) <= 100


def test_chunk_text_overlap_carries_words():
    words = [str(i) for i in range(200)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    last_words = chunks[0].split()[-10:]
    first_words_next = chunks[1].split()[:10]
    assert last_words == first_words_next


def test_chunk_text_empty_input():
    assert chunk_text("", 100, 10) == []


def test_chunk_text_raises_on_bad_overlap():
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("some text", chunk_size=10, overlap=10)


def test_chunk_text_single_chunk_when_short():
    text = "short text"
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == "short text"


def test_chunk_text_no_empty_chunks():
    text = " ".join(["word"] * 300)
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert all(c.strip() for c in chunks)


# ── build_user_prompt ─────────────────────────────────────────────────────────

def test_prompt_contains_question():
    prompt = build_user_prompt("What is the limit?", [])
    assert "What is the limit?" in prompt


def test_prompt_contains_source_and_chunk():
    prompt = build_user_prompt(
        "What is the limit?",
        [{"source": "policy.md", "chunk": 1, "score": 0.95, "text": "Limit is 50000."}],
    )
    assert "policy.md" in prompt
    assert "Chunk: 1" in prompt


def test_prompt_handles_empty_contexts():
    prompt = build_user_prompt("Any question?", [])
    assert "No relevant context" in prompt


def test_prompt_includes_score():
    prompt = build_user_prompt(
        "limit?",
        [{"source": "doc.md", "chunk": 2, "score": 0.87, "text": "some text"}],
    )
    assert "0.87" in prompt


def test_prompt_includes_context_text():
    prompt = build_user_prompt(
        "limit?",
        [{"source": "doc.md", "chunk": 1, "score": 0.9, "text": "Daily limit is INR 50000."}],
    )
    assert "Daily limit is INR 50000." in prompt


# ── SYSTEM_PROMPT ─────────────────────────────────────────────────────────────

def test_system_prompt_contains_anti_hallucination_rule():
    assert "never" in SYSTEM_PROMPT.lower() or "only" in SYSTEM_PROMPT.lower()


def test_system_prompt_contains_citation_instruction():
    assert "cit" in SYSTEM_PROMPT.lower()


def test_system_prompt_contains_prompt_injection_protection():
    assert "injection" in SYSTEM_PROMPT.lower() or "instructions" in SYSTEM_PROMPT.lower()


def test_system_prompt_not_empty():
    assert len(SYSTEM_PROMPT.strip()) > 100


# ── load_documents ────────────────────────────────────────────────────────────

def test_load_documents_raises_on_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nonexistent")


def test_load_documents_skips_unsupported_files(tmp_path):
    (tmp_path / "notes.csv").write_text("col1,col2")
    (tmp_path / "policy.md").write_text("# Policy\nSome content here.")
    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert docs[0]["source"] == "policy.md"


def test_load_documents_reads_md_file(tmp_path):
    content = "# ATM Policy\nDaily limit is INR 50,000."
    (tmp_path / "atm.md").write_text(content, encoding="utf-8")
    docs = load_documents(tmp_path)
    assert docs[0]["text"] == content


def test_load_documents_skips_empty_files(tmp_path):
    (tmp_path / "empty.md").write_text("   ", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert len(docs) == 0


def test_load_documents_reads_txt_file(tmp_path):
    (tmp_path / "note.txt").write_text("Some plain text.", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert docs[0]["text"] == "Some plain text."


def test_load_documents_returns_source_filename(tmp_path):
    (tmp_path / "fraud_policy.md").write_text("Fraud rules here.", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert docs[0]["source"] == "fraud_policy.md"


# ── rag package imports ───────────────────────────────────────────────────────

def test_rag_pipeline_importable():
    from rag.pipeline import RAGPipeline  # noqa: F401
    assert RAGPipeline is not None


def test_rag_index_importable():
    from rag.index import VectorIndex  # noqa: F401
    assert VectorIndex is not None


def test_vector_index_is_loaded_false_by_default(tmp_path):
    from rag.index import VectorIndex
    idx = VectorIndex(tmp_path / "test.index", tmp_path / "meta.json")
    assert idx.is_loaded is False


def test_vector_index_load_raises_when_missing(tmp_path):
    from rag.index import VectorIndex
    idx = VectorIndex(tmp_path / "nofile.index", tmp_path / "nofile.json")
    with pytest.raises(FileNotFoundError):
        idx.load()
