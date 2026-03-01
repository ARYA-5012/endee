"""
api/server.py
─────────────
FastAPI REST API for DevSearch — Semantic Code Intelligence.

Exposes all core operations as HTTP endpoints so the Streamlit frontend
(or any other client) can interact with Endee without importing core modules:

  GET   /api/health          → connection status
  GET   /api/indexes         → list indexed repos
  DELETE /api/indexes/{repo} → delete an index
  POST  /api/ingest          → index a GitHub repo
  POST  /api/search          → semantic code search
  POST  /api/ask             → RAG Q&A

Run:  uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import sys
import time
import logging
import traceback
from typing import List, Optional, Any

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.chunker      import chunk_file
from core.embedder     import embed_texts, embed_single
from core.endee_client import DevSearchDB
from core.repo_loader  import fetch_repo_files, parse_github_url
from core.rag          import ask as rag_ask

logger = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="DevSearch API",
    description="Semantic Code Intelligence — search any GitHub repo with natural language",
    version="1.0.0",
)

# CORS — allow Streamlit and any local frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler — always return JSON, never plain-text 500
from fastapi.responses import JSONResponse
from starlette.requests import Request

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.error(f"Unhandled exception: {exc}\n{tb}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_db() -> DevSearchDB:
    """Build a DevSearchDB from env vars."""
    return DevSearchDB(
        base_url=os.getenv("ENDEE_BASE_URL", "http://localhost:8080/api/v1"),
        auth_token=os.getenv("ENDEE_AUTH_TOKEN", ""),
    )


# ── Pydantic models ──────────────────────────────────────────────────────────
class IngestRequest(BaseModel):
    github_url: str = Field(..., description="Public GitHub repository URL")
    github_token: Optional[str] = Field(None, description="GitHub PAT (optional)")
    force: bool = Field(False, description="Force re-index if already exists")

class IngestResponse(BaseModel):
    repo: str
    files: int
    chunks: int
    vectors: int
    elapsed_seconds: float

class SearchRequest(BaseModel):
    repo: str = Field(..., description="Repo in 'owner/name' format")
    query: str = Field(..., description="Natural-language search query")
    top_k: int = Field(8, ge=1, le=50)
    language: Optional[str] = None

class SearchResultItem(BaseModel):
    chunk_id: str
    similarity: float
    file_path: str
    language: str
    repo: str
    function_name: str
    start_line: int
    end_line: int
    code: str

class SearchResponse(BaseModel):
    query: str
    repo: str
    count: int
    results: List[SearchResultItem]

class AskRequest(BaseModel):
    repo: str = Field(..., description="Repo in 'owner/name' format")
    question: str = Field(..., description="Natural-language question about the codebase")
    top_k: int = Field(6, ge=1, le=20)

class AskResponse(BaseModel):
    answer: str
    sources: List[SearchResultItem]

class IndexInfo(BaseModel):
    name: str
    repo_display: str
    num_vectors: Any = 0
    dimension: int = 384
    space_type: str = "cosine"
    precision: str = "INT8"

class HealthResponse(BaseModel):
    status: str
    endee_connected: bool
    index_count: int
    message: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check if the API and Endee are healthy."""
    try:
        db = _get_db()
        db.ping()
        indexes = db.list_indexes()
        return HealthResponse(
            status="ok",
            endee_connected=True,
            index_count=len(indexes),
            message=f"Endee connected with {len(indexes)} index(es)",
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            endee_connected=False,
            index_count=0,
            message=f"Cannot reach Endee: {e}",
        )


@app.get("/api/indexes", response_model=List[IndexInfo], tags=["Indexes"])
def list_indexes():
    """List all indexed repositories."""
    try:
        db = _get_db()
        indexes = db.list_indexes()
    except Exception as e:
        raise HTTPException(502, f"Cannot connect to Endee: {e}")

    result = []
    for idx in indexes:
        if isinstance(idx, dict):
            name = idx.get("name", "unknown")
            num_vecs = idx.get("num_vectors", idx.get("vector_count", 0))
            dimension = idx.get("dimension", 384)
            space_type = idx.get("space_type", "cosine")
            precision = idx.get("precision", "INT8")
        else:
            name = getattr(idx, "name", "unknown")
            num_vecs = getattr(idx, "num_vectors", getattr(idx, "vector_count", 0))
            dimension = getattr(idx, "dimension", 384)
            space_type = getattr(idx, "space_type", "cosine")
            precision = getattr(idx, "precision", "INT8")

        result.append(IndexInfo(
            name=name,
            repo_display=db.repo_name_from_index(name),
            num_vectors=num_vecs if isinstance(num_vecs, (int, float)) else 0,
            dimension=dimension,
            space_type=space_type,
            precision=precision,
        ))

    return result


@app.delete("/api/indexes/{owner}/{repo_name}", tags=["Indexes"])
def delete_index(owner: str, repo_name: str):
    """Delete an indexed repository."""
    repo = f"{owner}/{repo_name}"
    try:
        db = _get_db()
        success = db.delete_index(repo)
        if success:
            return {"status": "deleted", "repo": repo}
        raise HTTPException(500, f"Failed to delete index for '{repo}'")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Endee error: {e}")


@app.post("/api/ingest", response_model=IngestResponse, tags=["Ingestion"])
def ingest_repo(req: IngestRequest):
    """Index a GitHub repository into Endee."""
    # Validate URL
    try:
        owner, repo_name = parse_github_url(req.github_url)
        repo = f"{owner}/{repo_name}"
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Connect to Endee
    try:
        db = _get_db()
        db.ping()
    except Exception as e:
        raise HTTPException(502, f"Cannot connect to Endee: {e}")

    # Check existing
    already_exists = db.index_exists(repo)

    if already_exists and not req.force:
        raise HTTPException(
            409,
            f"'{repo}' is already indexed. Use force=true to re-index.",
        )

    if already_exists and req.force:
        db.delete_index(repo)

    try:
        db.create_index(repo, skip_exists_check=True)
    except Exception as e:
        logger.error(f"Failed to create index: {e}\n{traceback.format_exc()}")
        raise HTTPException(502, f"Failed to create Endee index: {e}")

    # Fetch files
    t0 = time.time()
    all_chunks = []
    file_count = 0

    try:
        for code_file, idx, total in fetch_repo_files(req.github_url, req.github_token):
            file_count += 1
            chunks = chunk_file(code_file)
            all_chunks.extend(chunks)
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch repository: {e}")

    if not all_chunks:
        raise HTTPException(
            422,
            "No code files found. The repo may be empty or use unsupported languages.",
        )

    # Embed
    texts = [c.code for c in all_chunks]
    embeddings = embed_texts(texts)

    # Upsert
    upserted = db.upsert_chunks(repo, all_chunks, embeddings)
    elapsed = time.time() - t0

    return IngestResponse(
        repo=repo,
        files=file_count,
        chunks=len(all_chunks),
        vectors=upserted,
        elapsed_seconds=round(elapsed, 1),
    )


@app.post("/api/search", response_model=SearchResponse, tags=["Search"])
def search_code(req: SearchRequest):
    """Semantic code search — find code by meaning, not keywords."""
    try:
        db = _get_db()
        q_vector = embed_single(req.query)
        results = db.search(
            repo=req.repo,
            query_vector=q_vector,
            top_k=req.top_k,
            language=req.language,
        )
    except RuntimeError as e:
        raise HTTPException(502, str(e))
    except Exception as e:
        raise HTTPException(500, f"Search failed: {e}")

    items = [
        SearchResultItem(
            chunk_id=r.chunk_id,
            similarity=r.similarity,
            file_path=r.file_path,
            language=r.language,
            repo=r.repo,
            function_name=r.function_name,
            start_line=r.start_line,
            end_line=r.end_line,
            code=r.code,
        )
        for r in results
    ]

    return SearchResponse(
        query=req.query,
        repo=req.repo,
        count=len(items),
        results=items,
    )


@app.post("/api/ask", response_model=AskResponse, tags=["RAG"])
def ask_codebase(req: AskRequest):
    """RAG Q&A — ask questions about an indexed codebase."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise HTTPException(
            400,
            "OPENROUTER_API_KEY not configured. Set it in your .env file.",
        )

    try:
        db = _get_db()
        answer, chunks = rag_ask(
            question=req.question,
            repo=req.repo,
            db=db,
            top_k=req.top_k,
            api_key=api_key,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(502, str(e))
    except Exception as e:
        raise HTTPException(500, f"RAG pipeline failed: {e}")

    sources = [
        SearchResultItem(
            chunk_id=c.chunk_id,
            similarity=c.similarity,
            file_path=c.file_path,
            language=c.language,
            repo=c.repo,
            function_name=c.function_name,
            start_line=c.start_line,
            end_line=c.end_line,
            code=c.code,
        )
        for c in chunks
    ]

    return AskResponse(answer=answer, sources=sources)
