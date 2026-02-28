"""
app/pages/1_🔍_Search.py
─────────────────────────
Semantic code search — embed a natural-language query with all-MiniLM-L6-v2,
query Endee for the nearest code chunks, display syntax-highlighted results.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.embedder     import embed_single
from core.endee_client import DevSearchDB

# Add app/ to path so styles module is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from styles import inject_css

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Search · DevSearch", page_icon="🔍", layout="wide")

inject_css(st)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🔍 Semantic Search")
st.markdown("Search any indexed repository using plain English — no keywords needed.")

# ── Sidebar config ────────────────────────────────────────────────────────────
endee_url   = st.session_state.get("endee_url",   os.getenv("ENDEE_BASE_URL", "http://localhost:8080/api/v1"))
endee_token = st.session_state.get("endee_token", os.getenv("ENDEE_AUTH_TOKEN", ""))

try:
    db      = DevSearchDB(base_url=endee_url, auth_token=endee_token or None)
    indexes = db.list_indexes()
except Exception as e:
    st.error(f"❌ Cannot connect to Endee: {e}")
    st.info("Start Endee and configure the URL in the Home page sidebar.")
    st.stop()

if not indexes:
    st.warning("No repositories indexed yet. Go to the **🏠 Home** page to index one.")
    st.stop()

# ── Controls ──────────────────────────────────────────────────────────────────
repo_names = []
for idx in indexes:
    name = idx.get("name", "") if isinstance(idx, dict) else getattr(idx, "name", "")
    if name:
        repo_names.append(db.repo_name_from_index(name))

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
# Check if an example query was clicked on previous run
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

# Auto-trigger search if loaded from an example click
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
    with st.spinner("Embedding query and searching Endee…"):
        try:
            q_vector = embed_single(query)
            results  = db.search(
                repo=selected_repo,
                query_vector=q_vector,
                top_k=top_k,
                language=lang_filter,
            )
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
                f'<span class="lang-badge">{r.language}</span>'
                f'<span class="file-path">📄 {r.file_path}</span>'
                + (f'  <span class="func-name">→ {r.function_name}()</span>' if r.function_name else "")
                + f'</div>'
                f'<div>'
                f'<span class="line-info">lines {r.start_line}–{r.end_line}</span>&nbsp;&nbsp;'
                f'<span class="sim-badge">similarity {r.similarity:.3f}</span>'
                f'</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Code with syntax highlighting (note: st.code starts at line 1)
            st.caption(f"↑ Lines {r.start_line}–{r.end_line} in original file")
            st.code(r.code, language=r.language, line_numbers=True)

            # Similar code button
            col_sim, col_ask, _ = st.columns([2, 2, 6])
            with col_sim:
                if st.button("🔁 Find similar", key=f"sim_{i}_{r.chunk_id}"):
                    st.session_state["sim_query"] = r.code[:300]
                    st.rerun()
            with col_ask:
                if st.button("🤖 Ask about this", key=f"ask_{i}_{r.chunk_id}"):
                    st.session_state["prefill_question"] = (
                        f"Explain the function `{r.function_name or r.file_path}` "
                        f"in `{r.file_path}` (lines {r.start_line}–{r.end_line})"
                    )
                    st.switch_page("pages/2_🤖_Ask.py")

            st.markdown("")

# ── Handle "Find Similar" re-query ───────────────────────────────────────────
if "sim_query" in st.session_state and not search_btn:
    sim_q = st.session_state.pop("sim_query")
    with st.spinner("Finding similar code…"):
        q_vector = embed_single(sim_q)
        results  = db.search(repo=selected_repo, query_vector=q_vector, top_k=top_k)
    if results:
        st.markdown("#### 🔁 Similar code chunks")
        for r in results[1:]:   # skip first (likely the same chunk)
            st.markdown(f"**`{r.file_path}`**  lines {r.start_line}–{r.end_line}  ·  similarity `{r.similarity:.3f}`")
            st.code(r.code, language=r.language)
