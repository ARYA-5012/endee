"""
app/pages/2_🤖_Ask.py
──────────────────────
RAG Q&A — retrieve relevant code from Endee, send to Gemini Flash,
stream the answer back with an evidence panel showing which chunks were used.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.endee_client import DevSearchDB
from core.rag          import ask

# Add app/ to path so styles module is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from styles import inject_css

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ask · DevSearch", page_icon="🤖", layout="wide")

inject_css(st)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🤖 Ask the Codebase")
st.markdown(
    "Ask any question about an indexed repository — Endee retrieves the most "
    "relevant code, which Gemini Flash uses to generate a grounded answer."
)

# ── Session config ────────────────────────────────────────────────────────────
endee_url      = st.session_state.get("endee_url",      os.getenv("ENDEE_BASE_URL", "http://localhost:8080/api/v1"))
endee_token    = st.session_state.get("endee_token",    os.getenv("ENDEE_AUTH_TOKEN", ""))
openrouter_key = st.session_state.get("openrouter_key", os.getenv("OPENROUTER_API_KEY", ""))

# ── Connect ───────────────────────────────────────────────────────────────────
try:
    db      = DevSearchDB(base_url=endee_url, auth_token=endee_token or None)
    indexes = db.list_indexes()
except Exception as e:
    st.error(f"❌ Cannot connect to Endee: {e}")
    st.stop()

if not indexes:
    st.warning("No repositories indexed yet. Go to the **🏠 Home** page to index one.")
    st.stop()

if not openrouter_key:
    st.warning(
        "⚠️ No OpenRouter API key found. Add it in the Home page sidebar or your `.env` file.\n\n"
        "Get a free key at [openrouter.ai](https://openrouter.ai)."
    )

# ── Repo selector ─────────────────────────────────────────────────────────────
repo_names = []
for idx in indexes:
    name = idx.get("name", "") if isinstance(idx, dict) else getattr(idx, "name", "")
    if name:
        repo_names.append(db.repo_name_from_index(name))

col_repo, col_k = st.columns([4, 1])
with col_repo:
    selected_repo = st.selectbox("Repository", repo_names)
with col_k:
    top_k = st.selectbox("Context chunks", [4, 6, 8, 10], index=1)

# ── Question input ────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill_question", "")
suggestion_prefill = st.session_state.pop("suggestion_question", "")
default_question = prefill or suggestion_prefill

question = st.text_area(
    "Your question",
    value=default_question,
    placeholder=(
        "How does this repo handle authentication?\n"
        "Explain the database connection pooling logic.\n"
        "What design patterns are used in the service layer?"
    ),
    height=100,
)

col_ask, col_clear = st.columns([2, 8])
with col_ask:
    ask_btn = st.button("🤖 Ask", type="primary", use_container_width=True)

# Auto-trigger if loaded from a suggestion click
if suggestion_prefill:
    ask_btn = True

# Suggested questions
with st.expander("💡 Suggested questions"):
    suggestions = [
        "How does error handling work in this codebase?",
        "Where and how is the database initialized?",
        "How does this project handle user authentication?",
        "What is the overall architecture of this application?",
        "How are API routes defined and organized?",
        "Explain the testing approach used in this repo.",
        "How is configuration managed across environments?",
        "What external services or APIs does this project integrate with?",
    ]
    cols = st.columns(2)
    for i, s in enumerate(suggestions):
        if cols[i % 2].button(s, key=f"sug_{i}", use_container_width=True):
            st.session_state["suggestion_question"] = s
            st.rerun()

# ── RAG pipeline ──────────────────────────────────────────────────────────────
if ask_btn and question.strip():
    if not openrouter_key:
        st.error("Add your OpenRouter API key in the Home page sidebar first.")
        st.stop()

    # Two-column layout: answer | evidence
    left, right = st.columns([3, 2])

    with left:
        st.markdown("### 💬 Answer")
        answer_placeholder = st.empty()
        answer_placeholder.info("Retrieving relevant code from Endee…")

    with right:
        st.markdown("### 📎 Evidence")
        evidence_placeholder = st.empty()

    try:
        answer, chunks = ask(
            question=question,
            repo=selected_repo,
            db=db,
            top_k=top_k,
            api_key=openrouter_key,
        )
    except Exception as e:
        answer_placeholder.error(f"Failed to get answer: {e}")
        st.stop()

    # ── Render answer ─────────────────────────────────────────────────────────
    with left:
        answer_placeholder.empty()
        st.markdown(answer)

    # ── Render evidence panel ─────────────────────────────────────────────────
    with right:
        evidence_placeholder.empty()
        for i, c in enumerate(chunks, start=1):
            fname = f" → `{c.function_name}`" if c.function_name else ""
            with st.expander(
                f"[{i}] `{c.file_path}`{fname}  ·  {c.similarity:.3f}",
                expanded=(i == 1),
            ):
                st.caption(f"Lines {c.start_line}–{c.end_line}  ·  {c.language}")
                st.code(c.code[:600], language=c.language)

    # ── Conversation history ──────────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state["history"] = []

    st.session_state["history"].append({
        "q": question, "a": answer, "repo": selected_repo, "chunks": len(chunks)
    })
    # Cap history to prevent unbounded growth
    st.session_state["history"] = st.session_state["history"][-20:]

# ── Conversation history ──────────────────────────────────────────────────────
history = st.session_state.get("history", [])
if history:
    st.markdown("---")
    st.markdown("### 📜 Session History")
    for item in reversed(history[:-1]):   # show previous, not the current one
        with st.expander(f"Q: {item['q'][:80]}…  · `{item['repo']}` · {item['chunks']} chunks"):
            st.markdown(item["a"])
