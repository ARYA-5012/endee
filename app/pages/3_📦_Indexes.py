"""
app/pages/3_📦_Indexes.py
──────────────────────────
Index management — view all indexed repositories and their stats,
delete stale indexes via the FastAPI backend.
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
st.set_page_config(page_title="Indexes · DevSearch", page_icon="📦", layout="wide")

inject_css(st)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 📦 Index Manager")
st.markdown("View and manage all GitHub repositories indexed in Endee.")

# ── API config ────────────────────────────────────────────────────────────────
API_BASE = st.session_state.get("api_url", os.getenv("DEVSEARCH_API_URL", "http://localhost:8000"))

# ── Fetch indexes ─────────────────────────────────────────────────────────────
try:
    resp = requests.get(f"{API_BASE}/api/indexes", timeout=5)
    resp.raise_for_status()
    indexes = resp.json()
except Exception as e:
    st.error(f"❌ Cannot connect to DevSearch API: {e}")
    st.stop()

# ── Summary ───────────────────────────────────────────────────────────────────
if not indexes:
    st.info("No repositories indexed yet. Go to the **🏠 Home** page to add one.")
    st.stop()

total_vecs = sum(int(i.get("num_vectors", 0) or 0) for i in indexes)

c1, c2, c3 = st.columns(3)
c1.metric("Total Indexes",  len(indexes))
c2.metric("Total Vectors",  f"{total_vecs:,}")
c3.metric("Embedding Model", "all-MiniLM-L6-v2")

st.markdown("---")

# ── Index cards ───────────────────────────────────────────────────────────────
for idx_info in indexes:
    name         = idx_info.get("name", "unknown")
    repo_display = idx_info.get("repo_display", name)
    num_vecs     = idx_info.get("num_vectors", 0)
    dimension    = idx_info.get("dimension", 384)
    space_type   = idx_info.get("space_type", "cosine")
    precision    = idx_info.get("precision", "INT8")

    num_vecs_display = f"{int(num_vecs):,}" if isinstance(num_vecs, (int, float)) else str(num_vecs)

    st.markdown(
        f'<div class="index-card">'
        f'<span class="index-name">🗂 {repo_display}</span>'
        f'<span class="badge-green">✓ indexed</span>'
        f'<div class="index-meta">'
        f'Vectors: <b>{num_vecs_display}</b> &nbsp;|&nbsp; '
        f'Dimensions: <b>{dimension}</b> &nbsp;|&nbsp; '
        f'Space: <b>{space_type}</b> &nbsp;|&nbsp; '
        f'Precision: <b>{precision}</b>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col_search, col_ask, col_delete, _ = st.columns([1.5, 1.5, 1.5, 6])

    with col_search:
        if st.button("🔍 Search", key=f"srch_{name}", use_container_width=True):
            st.session_state["selected_repo_search"] = repo_display
            st.switch_page("pages/1_🔍_Search.py")

    with col_ask:
        if st.button("🤖 Ask", key=f"ask_{name}", use_container_width=True):
            st.session_state["selected_repo_ask"] = repo_display
            st.switch_page("pages/2_🤖_Ask.py")

    with col_delete:
        if st.button("🗑 Delete", key=f"del_{name}", use_container_width=True):
            st.session_state[f"confirm_delete_{name}"] = True

    # Confirmation dialog
    if st.session_state.get(f"confirm_delete_{name}"):
        st.warning(f"Are you sure you want to delete **{repo_display}**? This cannot be undone.")
        yes, no = st.columns(2)
        with yes:
            if st.button("Yes, delete", key=f"yes_{name}", type="primary"):
                try:
                    # repo_display is "owner/repo" — split for the URL path
                    del_resp = requests.delete(
                        f"{API_BASE}/api/indexes/{repo_display}",
                        timeout=10,
                    )
                    if del_resp.status_code == 200:
                        st.success(f"Deleted {repo_display}")
                    else:
                        st.error(f"Delete failed: {del_resp.json().get('detail', del_resp.text)}")
                except Exception as e:
                    st.error(f"Delete failed: {e}")
                st.session_state.pop(f"confirm_delete_{name}", None)
                st.rerun()
        with no:
            if st.button("Cancel", key=f"no_{name}"):
                st.session_state.pop(f"confirm_delete_{name}", None)
                st.rerun()

    st.markdown("")


# ── API server info ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### ℹ️ DevSearch API")
st.code(
    f"API URL   : {API_BASE}\n"
    f"Health    : {API_BASE}/api/health\n"
    f"API Docs  : {API_BASE}/docs",
    language="text",
)
st.caption(
    "The FastAPI interactive docs (Swagger UI) are accessible at the URL above — "
    "you can test all endpoints directly from your browser."
)
