import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Initialised inside lifespan so model-download errors are caught and logged cleanly.
pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    try:
        pipeline = RAGPipeline()
        pipeline.load_index()
        logger.info("RAGPipeline ready — FAISS index loaded.")
    except FileNotFoundError:
        logger.warning("FAISS index missing. Run: python -m scripts.ingest")
    except Exception:
        logger.exception("Failed to initialise RAGPipeline.")
    yield


app = FastAPI(
    title="GenAI Banking Document Assistant",
    description="RAG-powered Q&A over banking policy documents.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=2000,
        examples=["What is the daily ATM withdrawal limit?"],
    )
    top_k: int = Field(default=4, ge=1, le=10)


class SourceItem(BaseModel):
    source: str
    chunk: int
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]


@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "index_loaded": pipeline is not None and pipeline.index.is_loaded,
    }


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask(request: AskRequest):
    if pipeline is None or not pipeline.index.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Run: python -m scripts.ingest then restart.",
        )
    try:
        logger.info("Question: %s", request.question)
        result = pipeline.ask(request.question, request.top_k)
        return AskResponse(
            question=request.question,
            answer=result["answer"],
            sources=result["sources"],
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /ask")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc
