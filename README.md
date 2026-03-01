# 🔍 DevSearch — Semantic Code Intelligence

> **Search any GitHub repository with plain English. No grep. No keywords. Just meaning.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red?logo=streamlit)](https://streamlit.io)
[![Endee](https://img.shields.io/badge/Vector_DB-Endee-6366f1)](https://github.com/endee-io/endee)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![OpenRouter](https://img.shields.io/badge/LLM-Gemini_Flash-orange)](https://openrouter.ai)

---

## 📌 Problem Statement

Every developer has spent hours navigating an unfamiliar codebase, asking:

- *"Where is authentication handled?"*
- *"How does this project connect to the database?"*
- *"What does the rate limiter actually do?"*

Traditional tools fail here. `grep` requires you to know the exact symbol name. IDE search is keyword-based. Documentation is often missing or stale.

**DevSearch solves this with vector similarity search.** You ask a question in plain English. DevSearch embeds it, queries Endee for the most semantically similar code chunks, and uses Gemini Flash to synthesize a grounded, cited answer — in seconds.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔎 **Semantic Search** | Natural language queries across an entire codebase using 384-dim vector similarity |
| 🤖 **RAG Q&A** | Ask questions; get LLM answers grounded in real retrieved code chunks |
| 🔁 **Similar Code Finder** | Click any result → find semantically identical patterns elsewhere |
| 🌐 **Any Public GitHub Repo** | Paste a URL, ingest in minutes — no clone needed |
| 📦 **Multi-Repo Index** | Index and switch between multiple repos in one UI |
| 🗂 **Index Manager** | View vector counts, stats, delete stale indexes |
| ⚡ **High Performance** | Endee handles up to 1B vectors on a single node with SIMD-optimized search |

---

## 🏗️ System Architecture

DevSearch uses a **3-tier architecture**: Streamlit frontend → FastAPI REST backend → Core pipeline → Endee vector DB.

```
┌──────────────────────────────────────────────────────────────────┐
│                     STREAMLIT FRONTEND (:8501)                   │
│  ┌──────────┐  ┌───────────────┐  ┌────────────┐  ┌──────────┐ │
│  │🏠 Ingest │  │ 🔍 Search     │  │ 🤖 Ask     │  │📦 Indexes│ │
│  └─────┬────┘  └──────┬────────┘  └─────┬──────┘  └────┬─────┘ │
└────────┼──────────────┼─────────────────┼───────────────┼───────┘
         │  HTTP/JSON   │                 │               │
         ▼              ▼                 ▼               ▼
┌──────────────────────────────────────────────────────────────────┐
│                   FASTAPI REST API (:8000)                        │
│                                                                  │
│  POST /api/ingest    GET /api/indexes    POST /api/search        │
│  POST /api/ask       DELETE /api/indexes/{repo}                  │
│  GET  /api/health                                                │
│                                                                  │
│  ┌──────────────┐  ┌─────────────┐  ┌───────────────┐           │
│  │ repo_loader  │  │  chunker    │  │   embedder    │           │
│  │ (GitHub API) │  │ (AST / SW)  │  │ (MiniLM-L6)   │           │
│  └──────────────┘  └─────────────┘  └───────────────┘           │
│         │                                    │                   │
│         │          ┌──────────────┐           │                   │
│         └─────────▶│  rag.py      │◀──────────┘                   │
│                    │ (OpenRouter) │                               │
│                    └──────────────┘                               │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                  ENDEE VECTOR DATABASE (:8080)                    │
│                                                                  │
│   create_index() · upsert(id, vector, meta) · query(vector, k)  │
│   384-dim cosine · INT8 quantized · HNSW · up to 1B vectors     │
└──────────────────────────────────────────────────────────────────┘
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System health + Endee connectivity check |
| `GET` | `/api/indexes` | List all indexed repositories |
| `DELETE` | `/api/indexes/{owner}/{repo}` | Delete an index |
| `POST` | `/api/ingest` | Index a GitHub repository |
| `POST` | `/api/search` | Semantic code search |
| `POST` | `/api/ask` | RAG Q&A with Gemini Flash |

Interactive API docs (Swagger UI) are available at `http://localhost:8000/docs`.

### Data Flow

1. **Ingest**: User pastes a GitHub URL → `repo_loader.py` fetches all code files via GitHub REST API (no `git clone`) → `chunker.py` splits into meaningful units → `embedder.py` produces 384-dim vectors → bulk-upserted into Endee.

2. **Search**: User types a natural-language query → embedded with the same model → Endee performs cosine nearest-neighbour search → top-k results returned with full metadata → displayed with syntax highlighting.

3. **Ask (RAG)**: Same retrieval as Search → top-k chunks assembled into a structured prompt → sent to Gemini Flash via OpenRouter → answer is grounded with file/function citations → evidence panel shows which chunks were used.

---

## ⚡ How Endee is Used

Endee replaces traditional vector stores (FAISS, ChromaDB, Pinecone) as the core retrieval engine. Here's what makes it the right choice for DevSearch:

### Index per Repository
Each GitHub repo gets its own Endee index, named `owner___repo` (triple-underscore separator, collision-safe). This keeps searches scoped and enables fast multi-repo switching without performance degradation.

```python
# endee_client.py
client.create_index(
    name="tiangolo___fastapi",
    dimension=384,           # matches all-MiniLM-L6-v2 output
    space_type="cosine",     # cosine similarity for normalized vectors
    precision=Precision.INT8 # quantized for low memory footprint
)
```

### Self-Contained Vectors
All retrieval context — file path, function name, line numbers, code snippet — is stored directly in Endee's `meta` field alongside each vector. Search results are fully self-contained; no secondary database lookup is required.

```python
index.upsert([{
    "id":     "a3f8c12d9b1e4f7a",   # stable SHA1 hash of (repo, path, line)
    "vector": [...],                 # 384-dim float32 list
    "meta": {
        "file_path":     "src/auth/jwt.py",
        "language":      "python",
        "function_name": "validate_token",
        "start_line":    45,
        "end_line":      78,
        "code":          "def validate_token(token: str) -> dict:\n    ..."
    }
}])
```

### Semantic Query
At search time, a single Endee `query()` call returns the k-nearest neighbours ranked by cosine similarity:

```python
results = index.query(vector=query_embedding, top_k=8)
# Each result: result.similarity, result.meta["code"], result.meta["file_path"] ...
```

### Scale Advantage
Endee is designed to handle **1 billion vectors on a single node** with HNSW indexing and SIMD acceleration (AVX2/AVX512/NEON). For a typical repo of ~5,000 chunks, searches complete in **< 5 ms** — fast enough for a real-time UI.

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Vector DB** | [Endee](https://github.com/endee-io/endee) | Stores & searches 384-dim code embeddings |
| **API Backend** | [FastAPI](https://fastapi.tiangolo.com) | REST API with auto-generated Swagger docs |
| **Embeddings** | `all-MiniLM-L6-v2` (HuggingFace) | 384-dim semantic vectors for code + queries |
| **Code Chunking** | Python `ast` + sliding window | AST-based for Python, windowed for all others |
| **LLM (RAG)** | Gemini Flash via OpenRouter | Generates grounded answers from retrieved code |
| **Frontend** | Streamlit | Interactive multi-page web UI |
| **Repo Loading** | GitHub REST API | Fetches code files without git clone |
| **Deployment** | Docker + Docker Compose | Both single-service and full-stack options |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose v2
- A free [OpenRouter API key](https://openrouter.ai) (for RAG Q&A)
- *(Optional)* A [GitHub personal access token](https://github.com/settings/tokens) (no scopes needed; increases API rate limit 60 → 5,000/hr)

---

## Option A: Endee in Docker, App Runs Locally *(Recommended for development)*

### 1. Clone the repository

```bash
git clone https://github.com/ARYA-5012/endee.git
cd endee
```

### 2. Start Endee

```bash
docker compose up -d
```

Endee is now running at `http://localhost:8080`.

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in at minimum:
```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the FastAPI backend

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at **http://localhost:8000/docs**

### 6. Launch the Streamlit UI

```bash
streamlit run app/main.py
```

Open **http://localhost:8501** in your browser. 🎉

---

## Option B: Full Stack with Docker Compose *(Recommended for production/demo)*

### 1. Clone & configure

```bash
git clone https://github.com/ARYA-5012/endee.git
cd endee
cp .env.example .env
# Edit .env — set OPENROUTER_API_KEY at minimum
```

### 2. Build & start everything

```bash
docker compose -f docker-compose.full.yml up --build
```

| Service | URL |
|---|---|
| DevSearch (Streamlit UI) | http://localhost:8501 |
| FastAPI Backend (Swagger) | http://localhost:8000/docs |
| Endee (Vector DB) | http://localhost:8080 |

To stop:
```bash
docker compose -f docker-compose.full.yml down
```

---

## 🖥️ Usage Guide

### 1. Index a Repository

Go to the **🏠 Home** page. Paste any public GitHub URL and click **⚡ Index**.

```
https://github.com/tiangolo/fastapi
https://github.com/pallets/flask
https://github.com/django/django
```

Indexing a typical medium-sized repo (~100 files) takes **60–120 seconds**.

Progress is shown in three phases:
- **Phase 1** — Fetching files from GitHub API
- **Phase 2** — Embedding with `all-MiniLM-L6-v2`
- **Phase 3** — Upserting vectors into Endee

### 2. Semantic Search

Go to **🔍 Search**. Select a repo and type any natural-language query:

- `"where is user authentication handled?"`
- `"database connection pooling"`
- `"error handling and logging"`
- `"how are API routes registered?"`

Results show:
- File path + function name
- Line numbers
- Similarity score (0–1)
- Syntax-highlighted code

### 3. Ask the Codebase (RAG)

Go to **🤖 Ask**. Type a question like:

- `"How does this project handle JWT tokens?"`
- `"Explain the overall architecture of the application"`
- `"What happens when a request fails validation?"`

The AI retrieves the most relevant code, reads it, and gives you a cited answer with references to specific files and functions. An **Evidence Panel** shows exactly which code chunks were used.

### 4. CLI Ingestion

For scripting or CI pipelines, use the CLI tool:

```bash
python scripts/ingest_repo.py https://github.com/tiangolo/fastapi \
    --token ghp_yourtoken \
    --force
```

---

## 📁 Project Structure

```
endee/
│
├── api/                            # FastAPI REST backend
│   ├── __init__.py
│   └── server.py                   # 6 REST endpoints + Swagger docs
│
├── app/                            # Streamlit frontend
│   ├── main.py                     # 🏠 Home — repo ingestion UI
│   ├── styles.py                   # Shared CSS (all pages)
│   └── pages/
│       ├── 1_🔍_Search.py          # Semantic search page
│       ├── 2_🤖_Ask.py             # RAG Q&A page
│       └── 3_📦_Indexes.py         # Index manager page
│
├── core/                           # Core intelligence pipeline
│   ├── __init__.py
│   ├── repo_loader.py              # GitHub API → CodeFile objects
│   ├── chunker.py                  # AST + sliding-window chunking
│   ├── embedder.py                 # HuggingFace sentence-transformers
│   ├── endee_client.py             # Endee SDK wrapper
│   └── rag.py                      # OpenRouter RAG pipeline
│
├── scripts/
│   └── ingest_repo.py              # CLI ingestion tool
│
├── .streamlit/
│   └── config.toml                 # Dark theme + server config
│
├── Dockerfile                      # Container image
├── docker-compose.yml              # Endee only (dev mode)
├── docker-compose.full.yml         # Full stack (Endee + API + UI)
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Configuration Reference

All configuration is via environment variables (`.env` file or Docker environment):

| Variable | Default | Description |
|---|---|---|
| `ENDEE_BASE_URL` | `http://localhost:8080/api/v1` | Endee server URL |
| `ENDEE_AUTH_TOKEN` | *(empty)* | Endee auth token (if set at startup) |
| `DEVSEARCH_API_URL` | `http://localhost:8000` | FastAPI backend URL (used by Streamlit) |
| `OPENROUTER_API_KEY` | *(required for Ask)* | OpenRouter API key |
| `OPENROUTER_MODEL` | `google/gemini-flash-1.5` | LLM model for RAG |
| `GITHUB_TOKEN` | *(optional)* | GitHub PAT for higher rate limits |

---

## 🧩 Chunking Strategy

| Language | Strategy | Details |
|---|---|---|
| Python | AST-based | `ast.parse()` extracts individual functions and classes with exact line ranges |
| JS / TS | Sliding window | 40-line windows, 10-line overlap, regex detects function boundaries |
| Java / Go / Rust / C++ / C | Sliding window | Same as above with language-specific function name extraction |
| All others | Sliding window | Generic fallback, 40-line windows |

Python gets AST-based chunking because Python's indentation makes the `ast` module reliable and free from brace-matching complexity. All other languages use sliding windows — simple, language-agnostic, and produce consistently useful chunks.

---

## 🤝 Contributing

Pull requests are welcome! Some ideas for contribution:

- Add tree-sitter support for richer multi-language AST chunking
- Add GitHub private repo support (OAuth flow)
- Add a diff-based re-indexing mode (only update changed files)
- Add export-to-JSON for search results
- Add semantic similarity clustering view

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

> Built with [Endee](https://github.com/endee-io/endee) · [Streamlit](https://streamlit.io) · [sentence-transformers](https://sbert.net) · [OpenRouter](https://openrouter.ai)
