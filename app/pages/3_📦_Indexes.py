"""
app/pages/3_📦_Indexes.py
──────────────────────────
Index management — view all indexed repositories and their stats,
delete stale indexes, and re-index a repo.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.endee_client import DevSearchDB

# Add app/ to path so styles module is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from styles import inject_css

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Indexes · DevSearch", page_icon="📦", layout="wide")

inject_css(st)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 📦 Index Manager")
st.markdown("View and manage all GitHub repositories indexed in Endee.")

# ── Connect ───────────────────────────────────────────────────────────────────
endee_url   = st.session_state.get("endee_url",   os.getenv("ENDEE_BASE_URL", "http://localhost:8080/api/v1"))
endee_token = st.session_state.get("endee_token", os.getenv("ENDEE_AUTH_TOKEN", ""))

try:
    db      = DevSearchDB(base_url=endee_url, auth_token=endee_token or None)
    indexes = db.list_indexes()
except Exception as e:
    st.error(f"❌ Cannot connect to Endee: {e}")
    st.stop()

# ── Summary ───────────────────────────────────────────────────────────────────
if not indexes:
    st.info("No repositories indexed yet. Go to the **🏠 Home** page to add one.")
    st.stop()

total_vecs = sum(
    int(i.get("num_vectors", i.get("vector_count", 0)) or 0)
    for i in indexes
)

c1, c2, c3 = st.columns(3)
c1.metric("Total Indexes",  len(indexes))
c2.metric("Total Vectors",  f"{total_vecs:,}")
c3.metric("Embedding Model", "all-MiniLM-L6-v2")

st.markdown("---")

# ── Index cards ───────────────────────────────────────────────────────────────
for idx_info in indexes:
    if isinstance(idx_info, dict):
        name       = idx_info.get("name", "unknown")
        num_vecs   = idx_info.get("num_vectors", idx_info.get("vector_count", "—"))
        dimension  = idx_info.get("dimension", 384)
        space_type = idx_info.get("space_type", "cosine")
        precision  = idx_info.get("precision", "INT8")
    else:
        name       = getattr(idx_info, "name", "unknown")
        num_vecs   = getattr(idx_info, "num_vectors", getattr(idx_info, "vector_count", "—"))
        dimension  = getattr(idx_info, "dimension", 384)
        space_type = getattr(idx_info, "space_type", "cosine")
        precision  = getattr(idx_info, "precision", "INT8")

    # Convert index name back to repo form
    repo_display = db.repo_name_from_index(name)

    # Safe formatting for num_vecs (may be string fallback)
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
                db.delete_index(repo_display)
                st.session_state.pop(f"confirm_delete_{name}", None)
                st.success(f"Deleted {repo_display}")
                st.rerun()
        with no:
            if st.button("Cancel", key=f"no_{name}"):
                st.session_state.pop(f"confirm_delete_{name}", None)
                st.rerun()

    st.markdown("")


# ── Endee server info ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### ℹ️ Endee Server")
st.code(
    f"Base URL  : {endee_url}\n"
    f"Auth      : {'enabled' if endee_token else 'disabled (open mode)'}\n"
    f"Dashboard : {endee_url.replace('/api/v1', '')}",
    language="text",
)
st.caption(
    "The Endee dashboard is accessible at the URL above — "
    "it shows index statistics, vector counts, and query performance metrics."
)
