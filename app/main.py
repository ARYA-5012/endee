"""
app/main.py  ·  DevSearch — Semantic Code Intelligence
───────────────────────────────────────────────────────
Home page: ingest a GitHub repository into Endee via the FastAPI backend.

Navigation (via Streamlit multi-page):
  🏠 Home (this page)   — ingest repos
  🔍 Search             — semantic search over any indexed repo
  🤖 Ask                — RAG Q&A powered by Gemini Flash
  📦 Indexes            — manage indexed repos
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

from styles import inject_css

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("DEVSEARCH_API_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DevSearch",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
inject_css(st)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    api_url = st.text_input(
        "API Base URL",
        value=API_BASE,
        help="URL of the DevSearch FastAPI server",
    )
    github_token = st.text_input(
        "GitHub Token (optional)",
        value=os.getenv("GITHUB_TOKEN", ""),
        type="password",
        help="Increases GitHub API rate limit from 60 → 5,000 req/hr",
    )
    openrouter_key = st.text_input(
        "OpenRouter API Key",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        type="password",
        help="Required for the Ask page. Get a free key at openrouter.ai",
    )

    # Persist to session state so other pages can read them
    st.session_state["api_url"]         = api_url
    st.session_state["github_token"]    = github_token
    st.session_state["openrouter_key"]  = openrouter_key

    st.divider()
    # ── API + Endee connection status ─────────────────────────────────────────
    try:
        resp = requests.get(f"{api_url}/api/health", timeout=15)
        data = resp.json()
        if data.get("endee_connected"):
            st.success(f"✅ API + Endee connected ({data['index_count']} indexes)")
            st.session_state["db_ok"] = True
        else:
            st.warning(f"⚠️ API running, Endee unreachable")
            st.caption(data.get("message", ""))
            st.session_state["db_ok"] = False
    except Exception as e:
        st.error("❌ Cannot reach DevSearch API")
        st.caption(f"Is the API running at {api_url}? → {e}")
        st.session_state["db_ok"] = False


# ── Hero section ──────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🔍 DevSearch</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Semantic Code Intelligence — search any GitHub repo with natural language</div>',
    unsafe_allow_html=True,
)

# Feature pills
for label in ["⚡ Powered by Endee", "🧠 all-MiniLM-L6-v2", "🤖 Gemini Flash RAG",
              "🐙 Any Public GitHub Repo", "🔎 384-dim Cosine Search", "🚀 FastAPI Backend"]:
    st.markdown(f'<span class="pill">{label}</span>', unsafe_allow_html=True)

st.markdown("---")


# ── Stats row ─────────────────────────────────────────────────────────────────
if st.session_state.get("db_ok"):
    try:
        resp = requests.get(f"{api_url}/api/indexes", timeout=5)
        indexes = resp.json()
        total_vectors = sum(int(i.get("num_vectors", 0) or 0) for i in indexes)
        c1, c2, c3, c4 = st.columns(4)
        for col, num, label in [
            (c1, len(indexes),    "Indexed Repos"),
            (c2, total_vectors,   "Vectors Stored"),
            (c3, 384,             "Embedding Dims"),
            (c4, "INT8",          "Precision"),
        ]:
            col.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-num">{num}</div>'
                f'<div class="stat-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("")
    except Exception:
        pass


# ── Ingestion form ────────────────────────────────────────────────────────────
st.markdown("### 📥 Index a Repository")
st.markdown(
    '<div class="info-box">Paste any public GitHub URL below. '
    "DevSearch will fetch the code, chunk it, embed it with "
    "<code>all-MiniLM-L6-v2</code>, and store the vectors in Endee — "
    "ready for semantic search in seconds.</div>",
    unsafe_allow_html=True,
)

col_url, col_btn = st.columns([5, 1])
with col_url:
    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/tiangolo/fastapi",
        label_visibility="collapsed",
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    ingest_btn = st.button("⚡ Index", type="primary", use_container_width=True)

force_reindex = st.checkbox("Force re-index (overwrite existing)", value=False)

if ingest_btn:
    if not repo_url.strip():
        st.warning("Please enter a GitHub repository URL.")
        st.stop()

    if not st.session_state.get("db_ok"):
        st.error("Endee is not reachable. Check the connection settings in the sidebar.")
        st.stop()

    # ── Call FastAPI ingest endpoint ──────────────────────────────────────────
    st.markdown(f"#### Indexing `{repo_url}`")
    progress_label = st.empty()
    progress_label.info("📥 Sending to API for ingestion... this may take a minute for large repos.")

    try:
        resp = requests.post(
            f"{api_url}/api/ingest",
            json={
                "github_url": repo_url,
                "github_token": st.session_state.get("github_token") or None,
                "force": force_reindex,
            },
            timeout=300,  # large repos may take a while
        )

        if resp.status_code == 409:
            st.info(f"✅ This repo is already indexed! Head to the Search or Ask pages. Use 'Force re-index' to overwrite.")
            st.stop()

        if resp.status_code != 200:
            error_detail = resp.json().get("detail", resp.text)
            st.error(f"Ingestion failed: {error_detail}")
            st.stop()

        result = resp.json()
        progress_label.empty()

        st.success(
            f"**{result['repo']}** is now indexed in Endee.\n\n"
            f"- {result['files']} files processed\n"
            f"- {result['chunks']} code chunks created\n"
            f"- {result['vectors']} vectors stored\n"
            f"- Completed in {result['elapsed_seconds']}s\n\n"
            f"Head to the **🔍 Search** or **🤖 Ask** pages to explore the code!"
        )

    except requests.exceptions.Timeout:
        st.error("Request timed out. The repo may be too large — try a smaller one.")
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach the API at {api_url}. Is it running?")
    except Exception as e:
        st.error(f"Unexpected error: {e}")


# ── How it works ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### ⚙️ How it works")

c1, c2, c3, c4 = st.columns(4)
for col, icon, title, desc in [
    (c1, "📥", "1. Fetch",    "GitHub API fetches all source files (no git clone needed)"),
    (c2, "✂️", "2. Chunk",   "Python uses AST; others use sliding window with overlap"),
    (c3, "🧠", "3. Embed",   "all-MiniLM-L6-v2 produces 384-dim vectors for each chunk"),
    (c4, "⚡", "4. Store",   "Endee stores vectors + metadata at 1B-vector scale"),
]:
    col.markdown(
        f"**{icon} {title}**  \n{desc}",
        unsafe_allow_html=False,
    )
