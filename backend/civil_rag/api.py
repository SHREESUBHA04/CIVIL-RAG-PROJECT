import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import shutil
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ── Import all modules ───────────────────────────────────────────
from civil_rag.vectorstore import load_index, build_index, search, embedder
from civil_rag.reranker import rerank
from civil_rag.query_expander import expand_and_search
from civil_rag.router import build_route_embeddings, route_query
from civil_rag.generator import generate_answer, handle_out_of_scope
from civil_rag.reflection import run_with_reflection
from civil_rag.dataset_analyzer import load_datasets, compute_statistics, answer_data_question
from civil_rag.ingest import ingest_documents

# ── Paths ────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).parent.parent / "data"
DATASETS_DIR = Path(__file__).parent.parent / "datasets"

# ─────────────────────────────────────────
# APP STARTUP
# ─────────────────────────────────────────

app = FastAPI(
    title="Civil Engineering RAG API",
    description="Agentic RAG system for civil engineering documents",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state (loaded once at startup) ────────────────────────
index        = None
chunks_store = None
route_vectors = None
datasets     = {}
stats        = {}

# Session memory: session_id → list of messages
conversation_store: Dict[str, List[Dict]] = {}


@app.on_event("startup")
async def startup():
    """
    Load all models and data at startup.
    This runs once when the server starts.
    Everything is kept in memory for fast responses.
    """
    global index, chunks_store, route_vectors, datasets, stats

    print("\n" + "═"*50)
    print("Starting Civil Engineering RAG API...")
    print("═"*50)

    # Load FAISS index
    try:
        index, chunks_store = load_index()
        print(f"✓ Vector index loaded: {index.ntotal} vectors")
    except FileNotFoundError:
        print("⚠ No index found. Upload documents to build one.")

    # Build semantic route vectors
    route_vectors = build_route_embeddings(embedder)
    print("✓ Semantic router ready")

    # Load datasets
    datasets = load_datasets(DATASETS_DIR)
    if datasets:
        stats = compute_statistics(datasets)
        print(f"✓ Datasets loaded: {list(datasets.keys())}")
    else:
        print("⚠ No datasets found in datasets/ folder")

    print("═"*50)
    print("API ready.\n")


# ─────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────

class AskRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str
    route: str
    reflection_passed: bool
    retried: bool
    confidence: Dict


# ─────────────────────────────────────────
# HELPER: COMPUTE HONEST CONFIDENCE
# ─────────────────────────────────────────

def compute_confidence(chunks: List[Dict]) -> Dict:
    """
    Compute honest confidence metrics from reranker scores.
    No fake percentages — only real measurements.
    """
    if not chunks:
        return {"level": "none", "best_score": 0, "chunks_found": 0}

    scores = [c.get("rerank_score", 0) for c in chunks]
    best   = max(scores)

    # Map reranker score to human-readable level
    # Cross-encoder scores: >5 = very relevant, 2-5 = relevant,
    # 0-2 = marginal, <0 = irrelevant (already filtered)
    if best >= 5.0:
        level = "high"
    elif best >= 2.0:
        level = "medium"
    else:
        level = "low"

    return {
        "level": level,
        "best_score": round(best, 4),
        "chunks_found": len(chunks),
    }


# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────

@app.get("/health")
async def health():
    """Check if API is running and what's loaded"""
    return {
        "status": "ok",
        "index_loaded": index is not None,
        "vectors": index.ntotal if index else 0,
        "datasets": list(datasets.keys()),
        "active_sessions": len(conversation_store),
    }


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Main question-answering endpoint.

    Flow:
    1. Get or create session
    2. Route query (semantic router)
    3. If DOCUMENT: expand → retrieve → rerank → generate → reflect
    4. If DATA: analyze datasets
    5. If OUT_OF_SCOPE: polite rejection
    6. Update conversation history
    7. Return answer with confidence metrics
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Get or create session
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in conversation_store:
        conversation_store[session_id] = []
    history = conversation_store[session_id]

    # Route the query
    route = route_query(query, embedder, route_vectors, verbose=False)

    # Handle each route
    if route == "OUT_OF_SCOPE":
        result = handle_out_of_scope()
        result["reflection"] = {"passed": True, "retried": False}

    elif route == "DATA":
        if not stats:
            result = {
                "answer": "No datasets are loaded. Add CSV files to the datasets/ folder.",
                "sources": [],
                "chunks": [],
                "reflection": {"passed": True, "retried": False}
            }
        else:
            result = answer_data_question(query, stats, history)
            result["reflection"] = {"passed": True, "retried": False}

    else:  # DOCUMENT
        if index is None:
            raise HTTPException(
                status_code=503,
                detail="No documents indexed yet. Please upload documents first."
            )
        result = run_with_reflection(
            query=query,
            index=index,
            chunks_store=chunks_store,
            embedder=embedder,
            route_vectors=route_vectors,
            route_fn=route_query,
            search_fn=search,
            expand_fn=expand_and_search,
            rerank_fn=rerank,
            generate_fn=generate_answer,
            handle_oos_fn=handle_out_of_scope,
            conversation_history=history,
        )

    # Update conversation history
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": result["answer"]})

    # Keep only last 10 exchanges
    if len(history) > 20:
        conversation_store[session_id] = history[-20:]

    # Compute confidence
    confidence = compute_confidence(result.get("chunks", []))

    return AskResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        session_id=session_id,
        route=route,
        reflection_passed=result.get("reflection", {}).get("passed", True),
        retried=result.get("reflection", {}).get("retried", False),
        confidence=confidence,
    )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document and rebuild the vector index.

    Accepts: PDF, DOCX, TXT
    Saves file to data/ folder, re-ingests all documents,
    rebuilds FAISS index.
    """
    global index, chunks_store

    # Validate file type
    allowed = {".pdf", ".docx", ".txt"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"File type {ext} not supported. Use PDF, DOCX, or TXT."
        )

    # Save file
    save_path = DATA_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    print(f"Saved: {save_path}")

    # Re-ingest all documents and rebuild index
    chunks = ingest_documents(DATA_DIR)
    if not chunks:
        raise HTTPException(status_code=500, detail="No text extracted from document")

    build_index(chunks)
    index, chunks_store = load_index()

    return {
        "message": f"'{file.filename}' uploaded and indexed successfully",
        "total_chunks": len(chunks_store),
        "total_vectors": index.ntotal,
    }


@app.delete("/clear_session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session"""
    if session_id in conversation_store:
        del conversation_store[session_id]
        return {"message": f"Session {session_id} cleared"}
    return {"message": "Session not found"}


@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "active_sessions": len(conversation_store),
        "session_ids": list(conversation_store.keys())
    }