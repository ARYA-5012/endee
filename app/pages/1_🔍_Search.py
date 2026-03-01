"""
app/pages/1_🔍_Search.py
─────────────────────────
Semantic code search — sends query to the FastAPI backend which embeds it,
queries Endee, and returns syntax-highlighted results.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Add app/ to path so styles module is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

from styles import inject_css

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Search · DevSearch", page_icon="🔍", layout="wide")

inject_css(st)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🔍 Semantic Search")
st.markdown("Search any indexed repository using plain English — no keywords needed.")

# ── API config ────────────────────────────────────────────────────────────────
API_BASE = st.session_state.get("api_url", os.getenv("DEVSEARCH_API_URL", "http://localhost:8000"))

# ── Fetch indexes ─────────────────────────────────────────────────────────────
try:
    resp = requests.get(f"{API_BASE}/api/indexes", timeout=5)
    resp.raise_for_status()
    indexes = resp.json()
except Exception as e:
    st.error(f"❌ Cannot connect to DevSearch API: {e}")
    st.info("Start the API server and configure the URL in the Home page sidebar.")
    st.stop()

if not indexes:
    st.warning("No repositories indexed yet. Go to the **🏠 Home** page to index one.")
    st.stop()

# ── Controls ──────────────────────────────────────────────────────────────────
repo_names = [idx.get("repo_display", idx.get("name", "")) for idx in indexes]

col_repo, col_k, col_lang = st.columns([3, 1, 1])

with col_repo:
    selected_repo = st.selectbox("Repository", repo_names)

with col_k:
    top_k = st.selectbox("Results", [5, 8, 10, 15, 20], index=1)

with col_lang:
    lang_filter = st.selectbox(
        "Language",
        ["All", "python", "javascript", "typescript", "java", "go",
         "rust", "cpp", "c", "csharp", "ruby", "kotlin", "swift"],
    )
    lang_filter = None if lang_filter == "All" else lang_filter

# ── Search bar ────────────────────────────────────────────────────────────────
default_query = st.session_state.pop("example_query", "")

col_q, col_btn = st.columns([6, 1])
with col_q:
    query = st.text_input(
        "Query",
        value=default_query,
        placeholder="e.g.  where is JWT authentication handled?",
        label_visibility="collapsed",
    )
with col_btn:
    search_btn = st.button("Search", type="primary", use_container_width=True)

if default_query:
    search_btn = True

# Example queries
with st.expander("💡 Example queries"):
    examples = [
        "where is user authentication handled?",
        "database connection setup",
        "error handling and exception management",
        "API rate limiting logic",
        "file upload and processing",
        "caching implementation",
        "password hashing and validation",
        "logging and monitoring setup",
    ]
    cols = st.columns(4)
    for i, ex in enumerate(examples):
        if cols[i % 4].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["example_query"] = ex
            st.rerun()

# ── Search execution ──────────────────────────────────────────────────────────
if search_btn and query.strip():
    with st.spinner("Searching via API…"):
        try:
            resp = requests.post(
                f"{API_BASE}/api/search",
                json={
                    "repo": selected_repo,
                    "query": query,
                    "top_k": top_k,
                    "language": lang_filter,
                },
                timeout=30,
            )
            if resp.status_code != 200:
                st.error(f"Search failed: {resp.json().get('detail', resp.text)}")
                st.stop()

            data = resp.json()
            results = data.get("results", [])
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

    if not results:
        st.info("No results found. Try rephrasing your query or removing the language filter.")
    else:
        st.markdown(f"**{len(results)} results** for `{query}` in `{selected_repo}`")
        st.markdown("---")

        for i, r in enumerate(results):
            # ── Result card ───────────────────────────────────────────────────
            st.markdown(
                f'<div class="result-card">'
                f'<div class="result-header">'
                f'<div>'
                f'<span class="lang-badge">{r["language"]}</span>'
                f'<span class="file-path">📄 {r["file_path"]}</span>'
                + (f'  <span class="func-name">→ {r["function_name"]}()</span>' if r.get("function_name") else "")
                + f'</div>'
                f'<div>'
                f'<span class="line-info">lines {r["start_line"]}–{r["end_line"]}</span>&nbsp;&nbsp;'
                f'<span class="sim-badge">similarity {r["similarity"]:.3f}</span>'
                f'</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Code with syntax highlighting
            st.caption(f"↑ Lines {r['start_line']}–{r['end_line']} in original file")
            st.code(r["code"], language=r["language"], line_numbers=True)

            # Action buttons
            col_sim, col_ask, _ = st.columns([2, 2, 6])
            with col_sim:
                if st.button("🔁 Find similar", key=f"sim_{i}_{r['chunk_id']}"):
                    st.session_state["sim_query"] = r["code"][:300]
                    st.rerun()
            with col_ask:
                if st.button("🤖 Ask about this", key=f"ask_{i}_{r['chunk_id']}"):
                    fname = r.get("function_name") or r["file_path"]
                    st.session_state["prefill_question"] = (
                        f"Explain the function `{fname}` "
                        f"in `{r['file_path']}` (lines {r['start_line']}–{r['end_line']})"
                    )
                    st.switch_page("pages/2_🤖_Ask.py")

            st.markdown("")

# ── Handle "Find Similar" re-query ───────────────────────────────────────────
if "sim_query" in st.session_state and not search_btn:
    sim_q = st.session_state.pop("sim_query")
    with st.spinner("Finding similar code…"):
        try:
            resp = requests.post(
                f"{API_BASE}/api/search",
                json={"repo": selected_repo, "query": sim_q, "top_k": top_k},
                timeout=30,
            )
            results = resp.json().get("results", [])
        except Exception:
            results = []
    if results:
        st.markdown("#### 🔁 Similar code chunks")
        for r in results[1:]:
            st.markdown(f"**`{r['file_path']}`**  lines {r['start_line']}–{r['end_line']}  ·  similarity `{r['similarity']:.3f}`")
            st.code(r["code"], language=r["language"])
