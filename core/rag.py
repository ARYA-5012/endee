"""
rag.py
──────
Retrieval-Augmented Generation pipeline for DevSearch.

Flow
────
  1. Embed the user's natural-language question.
  2. Retrieve the top-k most semantically similar code chunks from Endee.
  3. Build a structured prompt that includes those chunks as grounding context.
  4. Send the prompt to Gemini Flash via OpenRouter's OpenAI-compatible API.
  5. Return the response for rendering in the Streamlit UI.

The LLM is purposely instructed to:
  • Answer ONLY from the provided code context (no hallucinated file paths).
  • Cite which file / function each part of its answer comes from.
  • Admit when the context is insufficient.
"""

from __future__ import annotations

import os
from typing import List

import requests

from core.embedder import embed_single
from core.endee_client import DevSearchDB, SearchResult

# ── OpenRouter config ─────────────────────────────────────────────────────────
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL       = os.getenv("OPENROUTER_MODEL", "google/gemini-flash-1.5")
MAX_CONTEXT_CHUNKS  = 6      # number of retrieved chunks to include in the prompt
MAX_CODE_CHARS      = 1200   # chars per chunk in the prompt (token budget)


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are DevSearch — an expert code assistant with deep knowledge of software \
engineering. You answer questions strictly based on the code snippets provided \
in the context below. 

Rules:
1. Cite EVERY claim by mentioning the exact file path and function name shown \
in the context (e.g., "In `src/auth/jwt.py` → `validate_token`…").
2. If the context does not contain enough information to answer confidently, \
say so clearly — do NOT invent file paths or function names.
3. Use markdown code blocks (```language) when showing code in your answer.
4. Keep your answer concise but technically precise.
"""


# ── Public entry point ────────────────────────────────────────────────────────
def ask(
    question:    str,
    repo:        str,
    db:          DevSearchDB,
    top_k:       int = MAX_CONTEXT_CHUNKS,
    api_key:     str | None = None,
    model:       str = DEFAULT_MODEL,
) -> tuple[str, List[SearchResult]]:
    """
    Full RAG pipeline: embed → retrieve → generate.

    Returns
    -------
    (answer_text, retrieved_chunks)
        answer_text      : the LLM's markdown-formatted response
        retrieved_chunks : the SearchResult objects used as context
                          (shown in the UI as an evidence panel)
    """
    key = api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    # Step 1: Embed the question
    q_vector = embed_single(question)

    # Step 2: Retrieve relevant chunks from Endee
    chunks = db.search(repo=repo, query_vector=q_vector, top_k=top_k)
    if not chunks:
        return (
            "⚠️ No relevant code was found in the index for this repository. "
            "Try re-indexing or rephrasing your question.",
            [],
        )

    # Step 3: Build the prompt
    context_block = _build_context(chunks)
    user_message  = f"### Code Context\n\n{context_block}\n\n### Question\n{question}"

    # Step 4: Call OpenRouter
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/devsearch",
        "X-Title":       "DevSearch",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": user_message},
        ],
        "temperature": 0.2,
        "max_tokens":  1024,
    }

    try:
        resp = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "OpenRouter request timed out after 60s. The model may be overloaded — try again."
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not connect to OpenRouter. Check your internet connection."
        )

    if resp.status_code == 429:
        raise RuntimeError(
            "OpenRouter rate limit exceeded. Wait a minute and try again, "
            "or switch to a different model in your .env file."
        )
    if resp.status_code != 200:
        raise RuntimeError(
            f"OpenRouter returned HTTP {resp.status_code}: {resp.text[:200]}"
        )

    data = resp.json()
    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(
            f"Unexpected response format from OpenRouter: {str(data)[:300]}"
        )

    return answer, chunks


# ── Helpers ───────────────────────────────────────────────────────────────────
def _build_context(chunks: List[SearchResult]) -> str:
    """Format retrieved chunks into a clear, numbered context block."""
    parts: List[str] = []
    for i, c in enumerate(chunks, start=1):
        fname = f" → `{c.function_name}`" if c.function_name else ""
        header = (
            f"[{i}] **`{c.file_path}`**{fname}  "
            f"(lines {c.start_line}–{c.end_line}, similarity {c.similarity:.2f})"
        )
        code_block = f"```{c.language}\n{c.code[:MAX_CODE_CHARS]}\n```"
        parts.append(f"{header}\n{code_block}")
    return "\n\n---\n\n".join(parts)
