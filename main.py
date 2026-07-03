"""
main.py
-------
FastAPI backend exposing the RAG Study Assistant as a set of HTTP endpoints.

Endpoints:
  POST /ingest          -> re-index all PDFs in data/course_pdfs
  POST /ask             -> ask a question, get a grounded answer + citations
  GET  /health          -> simple health check
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_engine import RAGEngine

app = FastAPI(title="RAG-Powered AI Study Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RAGEngine()


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest():
    try:
        n_chunks = engine.ingest_directory()
        return {"status": "success", "chunks_indexed": n_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = engine.answer(req.question, k=req.top_k or 4)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("BACKEND_HOST", "0.0.0.0"),
        port=int(os.getenv("BACKEND_PORT", 8000)),
        reload=True,
    )
