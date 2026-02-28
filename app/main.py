"""
app/main.py  ·  DevSearch — Semantic Code Intelligence
───────────────────────────────────────────────────────
Home page: ingest a GitHub repository into Endee.

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.chunker      import chunk_file
from core.embedder     import embed_texts
from core.endee_client import DevSearchDB
from core.repo_loader  import fetch_repo_files, parse_github_url

from styles import inject_css

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
    endee_url = st.text_input(
        "Endee Base URL",
        value=os.getenv("ENDEE_BASE_URL", "http://localhost:8080/api/v1"),
        help="URL of your running Endee server",
    )
    endee_token = st.text_input(
        "Endee Auth Token (optional)",
        value=os.getenv("ENDEE_AUTH_TOKEN", ""),
        type="password",
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
    st.session_state["endee_url"]       = endee_url
    st.session_state["endee_token"]     = endee_token
    st.session_state["github_token"]    = github_token
    st.session_state["openrouter_key"]  = openrouter_key

    st.divider()
    # ── Endee connection status ───────────────────────────────────────────────
    try:
        db = DevSearchDB(base_url=endee_url, auth_token=endee_token or None)
        db.ping()  # Verify Endee is actually reachable
        indexes = db.list_indexes()
        st.success(f"✅ Endee connected  ({len(indexes)} index{'es' if len(indexes) != 1 else ''})")
        st.session_state["db_ok"] = True
    except Exception as e:
        st.error("❌ Cannot reach Endee")
        st.caption(str(e))
        st.session_state["db_ok"] = False


# ── Hero section ──────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🔍 DevSearch</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Semantic Code Intelligence — search any GitHub repo with natural language</div>',
    unsafe_allow_html=True,
)

# Feature pills
for label in ["⚡ Powered by Endee", "🧠 all-MiniLM-L6-v2", "🤖 Gemini Flash RAG",
              "🐙 Any Public GitHub Repo", "🔎 384-dim Cosine Search"]:
    st.markdown(f'<span class="pill">{label}</span>', unsafe_allow_html=True)

st.markdown("---")


# ── Stats row ─────────────────────────────────────────────────────────────────
if st.session_state.get("db_ok"):
    try:
        indexes = db.list_indexes()
        total_vectors = sum(
            int(i.get("num_vectors", i.get("vector_count", 0)) or 0)
            for i in indexes
        )
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

    try:
        owner, repo_name = parse_github_url(repo_url)
        repo = f"{owner}/{repo_name}"
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # ── Check existing index ──────────────────────────────────────────────────
    db = DevSearchDB(
        base_url=st.session_state["endee_url"],
        auth_token=st.session_state["endee_token"] or None,
    )

    already_exists = db.index_exists(repo)  # single API call

    if already_exists and not force_reindex:
        st.info(f"✅ **{repo}** is already indexed! Head to the Search or Ask pages.")
        st.stop()

    if already_exists and force_reindex:
        db.delete_index(repo)

    db.create_index(repo, skip_exists_check=already_exists and force_reindex)

    # ── Progress UI ───────────────────────────────────────────────────────────
    st.markdown(f"#### Indexing `{repo}`")
    phase_label = st.empty()
    progress_bar = st.progress(0.0)
    status_col1, status_col2, status_col3 = st.columns(3)
    files_metric   = status_col1.empty()
    chunks_metric  = status_col2.empty()
    vectors_metric = status_col3.empty()

    phase_label.markdown("**Phase 1 / 3 — Fetching files from GitHub…**")
    files_metric.metric("Files", 0)
    chunks_metric.metric("Chunks", 0)
    vectors_metric.metric("Vectors", 0)

    # Collect files + chunks
    t0 = time.time()
    all_chunks = []
    file_count = 0

    gh_token = st.session_state.get("github_token") or None

    try:
        for code_file, idx, total in fetch_repo_files(repo_url, gh_token):
            file_count += 1
            chunks = chunk_file(code_file)
            all_chunks.extend(chunks)
            progress_bar.progress(min(idx / total * 0.4, 0.4))
            files_metric.metric("Files",  file_count)
            chunks_metric.metric("Chunks", len(all_chunks))
    except Exception as e:
        st.error(f"Failed to fetch repository: {e}")
        st.stop()

    if not all_chunks:
        st.warning("No code files found. The repo may be empty or use unsupported languages.")
        st.stop()

    # Embed
    phase_label.markdown("**Phase 2 / 3 — Embedding with all-MiniLM-L6-v2…**")
    progress_bar.progress(0.4)
    texts      = [c.code for c in all_chunks]
    embeddings = embed_texts(texts)
    progress_bar.progress(0.7)

    # Upsert into Endee
    phase_label.markdown("**Phase 3 / 3 — Storing vectors in Endee…**")
    upserted = db.upsert_chunks(repo, all_chunks, embeddings)
    progress_bar.progress(1.0)

    elapsed = time.time() - t0
    vectors_metric.metric("Vectors", upserted)
    phase_label.markdown(f"✅ **Done in {elapsed:.1f}s!**")

    st.success(
        f"**{repo}** is now indexed in Endee.\n\n"
        f"- {file_count} files processed\n"
        f"- {len(all_chunks)} code chunks created\n"
        f"- {upserted} vectors stored\n\n"
        f"Head to the **🔍 Search** or **🤖 Ask** pages to explore the code!"
    )


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
