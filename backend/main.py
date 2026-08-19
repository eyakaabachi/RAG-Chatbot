import logging
import time
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from rag_pipeline import DocumentIndex, generate_answer
from schemas import AnswerContract

# ---------------------------------------------------------------------------
# Structured logging: JSON lines, one per request, so a real deployment
# could ship these straight to a log aggregator without reformatting.
# This is the same instinct as Pattern 2 in the extraction-errors article:
# every step should be replayable, not just trusted.
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("doc-chatbot")

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DOCS_DIR = BASE_DIR / "data" / "sample_docs"
FRONTEND_DIR = BASE_DIR / "frontend"

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Document Chatbot Prototype")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

index = DocumentIndex()

# In-memory metrics. No external monitoring service, so this is deliberately
# minimal: enough to show request volume, latency, and answer quality trends
# without adding a paid dependency to a free-tier deployment.
metrics = {
    "requests_total": 0,
    "answer_found_total": 0,
    "errors_total": 0,
    "latency_ms_sum": 0.0,
}


@app.on_event("startup")
def load_sample_corpus():
    if SAMPLE_DOCS_DIR.exists():
        index.load_folder(str(SAMPLE_DOCS_DIR))
    logger.info('{"event": "startup", "chunks_indexed": %d}' % len(index.chunks))


class AskRequest(BaseModel):
    question: str


@app.post("/api/ask", response_model=AnswerContract)
@limiter.limit("10/minute")  # protects the free Groq/embedding quota from abuse
def ask(request: Request, req: AskRequest):
    start = time.perf_counter()
    metrics["requests_total"] += 1

    try:
        retrieved = index.search(req.question)
        answer = generate_answer(req.question, retrieved)
    except Exception:
        metrics["errors_total"] += 1
        logger.exception('{"event": "ask_failed"}')
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000
    metrics["latency_ms_sum"] += elapsed_ms
    if answer.answer_found:
        metrics["answer_found_total"] += 1

    logger.info(
        '{"event": "ask", "question_len": %d, "language": "%s", '
        '"answer_found": %s, "complete": %s, "confidence": %.2f, '
        '"top_score": %.3f, "latency_ms": %.1f}'
        % (
            len(req.question),
            answer.language_detected or "unknown",
            str(answer.answer_found).lower(),
            str(answer.complete_answer_found).lower(),
            answer.confidence,
            retrieved[0].final_score if retrieved else 0.0,
            elapsed_ms,
        )
    )
    return answer


@app.post("/api/upload")
@limiter.limit("5/minute")
async def upload(request: Request, file: UploadFile = File(...)):
    content = (await file.read()).decode("utf-8", errors="ignore")
    doc_id = Path(file.filename).stem
    index.add_document(doc_id, content)
    logger.info(
        '{"event": "upload", "doc_id": "%s", "chunks_total": %d}' % (doc_id, len(index.chunks))
    )
    return {"doc_id": doc_id, "chunks_added": len(index.chunks)}


@app.get("/api/health")
def health():
    return {"status": "ok", "chunks_indexed": len(index.chunks)}


@app.get("/api/metrics")
def get_metrics():
    total = metrics["requests_total"]
    return JSONResponse(
        {
            "requests_total": total,
            "errors_total": metrics["errors_total"],
            "answer_found_rate": round(metrics["answer_found_total"] / total, 3) if total else None,
            "avg_latency_ms": round(metrics["latency_ms_sum"] / total, 1) if total else None,
        }
    )


# Serve the chat frontend as static files, single-service deploy
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
