# 🔍 DevSearch — Semantic Code Intelligence

> **Search any GitHub repository with plain English. No grep. No keywords. Just meaning.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red?logo=streamlit)](https://streamlit.io)
[![Endee](https://img.shields.io/badge/Vector_DB-Endee-6366f1)](https://github.com/endee-io/endee)
[![OpenRouter](https://img.shields.io/badge/LLM-Gemma_2_9B_IT-orange)](https://openrouter.ai)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Project Overview & Problem Statement

Every developer has spent hours navigating an unfamiliar codebase, asking:
- *"Where is authentication handled?"*
- *"How does this project connect to the database?"*
- *"What does the rate limiter actually do?"*

Traditional tools fail here. `grep` requires you to know the exact symbol name. IDE search is strictly keyword-based. Documentation is often missing or stale.

**DevSearch solves this with semantic vector search.** You ask a question in plain English. DevSearch embeds your query, searches **Endee** (a high-performance vector database) for the most semantically similar code chunks, and uses **Gemini / Gemma** via OpenRouter to synthesize a grounded, cited answer — in seconds.

---

## 💡 Practical Use Cases Demonstrated

This project serves as a showcase of modern AI-powered developer tooling, demonstrating two core practical applications:

### A. Semantic Search
Instead of guessing variable names, developers can describe functionality. 
- **Example Query:** *"how are API routes registered?"*
- **Result:** DevSearch bypasses the lack of exact keywords, semantically matching the query against code chunks (e.g., finding the FastAPI `include_router` functions) and highlighting the exact files and lines of code.

### B. RAG (Retrieval-Augmented Generation)
We take Semantic Search a step further to provide synthesis.
- **Example Query:** *"Explain the database connection pooling logic."*
- **Result:** DevSearch retrieves the top-K relevant code chunks and injects them into a strict prompt. The LLM reads the *actual* project code and generates an explanation, citing the exact file paths and function names so the developer can trust the answer. No hallucinations.

---

## 🚀 Recommendations for Extensions

The architecture of DevSearch makes it a perfect foundation for building more advanced systems. Here are recommendations for scaling this project:

### A. Agentic AI Workflows
DevSearch currently acts as an oracle. The next evolutionary step is making it an **Agent**:
1. **Automated Issue Resolution:** An autonomous agent uses the `/api/search` endpoint to find the bug's location, reads the code, and drafts a Pull Request.
2. **Codebase Migration Assistant:** Give an agent a goal (e.g., "Migrate all Flask routes to FastAPI"). It queries the `/api/indexes` for the current routes, plans the migration, and writes the new code iteratively.

### B. Similar AI Applications (Vector Search Core)
The ingestion pipeline (GitHub → Chunker → Embedder → Endee) can be adapted for:
- **Internal Enterprise Knowledge Bases:** Connect to Slack, Jira, and Confluence APIs to ingest company data so employees can ask *"Why did we change the billing schema last year?"*
- **Customer Support Copilots:** Index Zendesk tickets and product documentation to instantly suggest historically successful resolutions to live agents.

---

## 🏗️ System Design & Technical Approach

DevSearch uses a **3-tier architecture**: Streamlit frontend → FastAPI REST backend → Endee vector DB.

```text
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
│   384-dim cosine · float32 precision · up to 1B vectors         │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow
1. **Ingest**: User pastes a GitHub URL → `repo_loader.py` fetches code files via GitHub REST API (no `git clone`) → `chunker.py` splits into structural units (AST for Python, sliding-window for others) → `embedder.py` produces 384-dim vectors via `all-MiniLM-L6-v2`.
2. **Upsert**: Vectors + source code metadata are bulk-upserted into Endee.
3. **Retrieval (Search/Ask)**: User queries are embedded → Endee performs cosine nearest-neighbour search → FastAPI returns syntax-highlighted chunks or sends them to OpenRouter for RAG synthesis.

---

## ⚡ How Endee is Used

Endee replaces traditional vector stores (FAISS, ChromaDB, Pinecone) as the core retrieval engine. Here is how it's integrated:

### 1. Index per Repository
Each GitHub repo gets its own isolated Endee index, named `owner___repo` (e.g., `arya__5012___endee`). This keeps searches scoped and prevents cross-repository pollution.

```python
client.create_index(
    name="tiangolo___fastapi",
    dimension=384,           # matches all-MiniLM-L6-v2 output
    space_type="cosine",     # cosine similarity for normalized vectors
    precision="float32"      
)
```

### 2. Self-Contained Retrieval Data
All retrieval context — file path, function name, line numbers, and the raw code snippet — is stored directly in Endee's `meta` field alongside each vector. 

```python
index.upsert([{
    "id":     "a3f8c12d9b1e4f7a",   # stable ID
    "vector": [...],                 # 384-dim float32 list
    "meta": {
        "file_path":     "src/auth/jwt.py",
        "language":      "python",
        "code":          "def validate_token(token: str) -> dict:\n    ..."
    }
}])
```

### 3. Hyper-fast Semantic Queries
At query time, the system performs an ultra-low latency nearest neighbor search. Because the `meta` field holds the code snippet, no external PostgreSQL/MongoDB lookup is required.

```python
# Returns top 8 chunks instantly
results = index.query(vector=query_embedding, top_k=8) 
```

---

## 🚀 Setup and Execution Instructions

### Prerequisites
- Python 3.11+
- Docker & Docker Compose v2 🐳
- A free [OpenRouter API key](https://openrouter.ai/) for RAG Q&A
- *(Optional)* A [GitHub Personal Access Token](https://github.com/settings/tokens) (Raises API limit from 60 → 5,000 requests/hr)

### 💻 Option A: Full Stack with Docker Compose (Recommended)

This spins up the Endee DB, FastAPI backend, and Streamlit UI in a single command.

1. **Clone the repository & configure `.env`:**
   ```bash
   git clone https://github.com/ARYA-5012/endee.git
   cd endee
   cp .env.example .env
   ```
   Edit `.env` and add your `OPENROUTER_API_KEY`.

2. **Build and start the stack:**
   ```bash
   docker compose -f docker-compose.full.yml up --build -d
   ```

3. **Access the Application:**
   * **Streamlit UI:** [http://localhost:8501](http://localhost:8501)
   * **FastAPI Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
   * **Endee Server:** `http://localhost:8080`

### 🔧 Option B: Local Python Dev Mode

Run Endee via Docker, but run the API and Streamlit directly on your host machine for rapid development.

1. **Start Endee Database:**
   ```bash
   docker compose up -d
   ```

2. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   ```bash
   cp .env.example .env
   # Add your OPENROUTER_API_KEY to .env
   ```

4. **Start the FastAPI Backend (Terminal 1):**
   ```bash
   uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Start the Streamlit Frontend (Terminal 2):**
   ```bash
   streamlit run app/main.py
   ```

Navigate to [http://localhost:8501](http://localhost:8501) in your browser to begin indexing repositories!

---

## 🤝 Contributing
Pull requests are welcome! Potential areas for contribution:
- Add `tree-sitter` support for multi-language AST chunking.
- Add GitHub private repo support via OAuth.
- Add a diff-based re-indexing mode for massive mono-repos.

---

## 👨‍💻 Author

<table>
<tr>
<td align="center">
<strong>Arya Yadav</strong><br>
Bennett University<br>
<a href="mailto:aryayadav5012@gmail.com">📧 Email</a> |
<a href="https://github.com/ARYA-5012">🐙 GitHub</a>
</td>
</tr>
</table>

---
## 📄 License
MIT License — see [LICENSE](LICENSE) for details.
