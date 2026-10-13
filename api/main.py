from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rag.pipeline import RAGPipeline

app = FastAPI(title="GenAI Banking Document Assistant API", version="1.0.0")
pipeline = RAGPipeline()
try:
    pipeline.load_index()
except FileNotFoundError:
    pass

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)

@app.get("/health")
def health():
    return {"status": "ok", "index_loaded": pipeline.index.index is not None}

@app.post("/ask")
def ask(request: AskRequest):
    if pipeline.index.index is None:
        raise HTTPException(status_code=503, detail="Index is not loaded. Run python scripts/ingest.py first.")
    try:
        return pipeline.ask(request.question, request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
