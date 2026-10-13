import logging
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    raise ValueError(f"Unsupported file type: {suffix}")


def load_documents(directory: Path) -> list[dict]:
    if not directory.exists():
        raise FileNotFoundError(f"Document directory not found: {directory}")
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                text = load_file(path).strip()
                if text:
                    records.append({"source": path.name, "text": text})
                    logger.info("Loaded: %s (%d chars)", path.name, len(text))
                else:
                    logger.warning("Skipped empty file: %s", path.name)
            except Exception as exc:
                logger.warning("Could not load %s: %s", path.name, exc)
    return records


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    Args:
        text: Input text to chunk.
        chunk_size: Maximum number of words per chunk.
        overlap: Number of words to overlap between consecutive chunks.

    Returns:
        List of non-empty text chunks.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})."
        )
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
    return chunks
