from pathlib import Path
from pypdf import PdfReader

def load_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported file type: {suffix}")

def load_documents(directory: Path):
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"}:
            text = load_file(path).strip()
            if text:
                records.append({"source": path.name, "text": text})
    return records

def chunk_text(text: str, chunk_size=900, overlap=150):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    words = text.split()
    chunks = []
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
