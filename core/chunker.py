"""
chunker.py
──────────
Splits source files into semantically meaningful chunks ready for embedding.

Strategy
--------
• Python  → AST-based: extract individual functions and classes with ast module.
• Others  → Sliding window (40 lines, 10-line overlap) that tries to align
            chunk boundaries with function/class declaration lines.

Each chunk carries metadata (file path, language, function name, line range,
and the raw code text) that will later be stored in Endee's meta field so
results are self-contained — no extra lookup required at query time.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from typing import List

from core.repo_loader import CodeFile

# ── Chunk size constants ──────────────────────────────────────────────────────
WINDOW_LINES   = 40     # lines per sliding-window chunk
OVERLAP_LINES  = 10     # lines of overlap between consecutive chunks
MAX_CHUNK_CHARS = 3000  # hard cap — keep embeddings meaningful


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class CodeChunk:
    chunk_id:      str            # stable unique ID  (used as Endee vector ID)
    file_path:     str
    language:      str
    repo:          str
    code:          str            # the actual source text
    start_line:    int
    end_line:      int
    function_name: str = ""       # best-effort — empty for windowed chunks


# ── Public entry point ────────────────────────────────────────────────────────
def chunk_file(file: CodeFile) -> List[CodeChunk]:
    """Return a list of CodeChunks for a single CodeFile."""
    if file.language == "python":
        chunks = _chunk_python(file)
    else:
        chunks = _chunk_sliding_window(file)

    # Assign stable IDs
    for c in chunks:
        c.chunk_id = _make_id(c)

    return chunks


# ── Python: AST-based chunking ────────────────────────────────────────────────
def _chunk_python(file: CodeFile) -> List[CodeChunk]:
    lines = file.content.splitlines()
    chunks: List[CodeChunk] = []

    try:
        tree = ast.parse(file.content)
    except SyntaxError:
        # Fall back to sliding window if the file won't parse
        return _chunk_sliding_window(file)

    # Walk top-level nodes only (avoids nested functions duplicating content)
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not hasattr(node, "end_lineno"):
            continue  # Python < 3.8 — unlikely but safe

        start = node.lineno - 1          # 0-indexed
        end   = node.end_lineno          # exclusive upper bound

        code = "\n".join(lines[start:end])
        if not code.strip() or len(code) < 20:
            continue

        # Truncate very large nodes (e.g. 600-line classes) to keep embeddings useful
        code = code[:MAX_CHUNK_CHARS]

        chunks.append(CodeChunk(
            chunk_id="",
            file_path=file.path,
            language=file.language,
            repo=file.repo,
            code=code,
            start_line=node.lineno,
            end_line=node.end_lineno,
            function_name=node.name,
        ))

    # If the file has no top-level defs (e.g. it's all module-level script code)
    # fall back to a single chunk of the whole file or a sliding window.
    if not chunks:
        return _chunk_sliding_window(file)

    return chunks


# ── Sliding-window chunking (all other languages) ────────────────────────────
def _chunk_sliding_window(file: CodeFile) -> List[CodeChunk]:
    lines = file.content.splitlines()
    if not lines:
        return []

    chunks: List[CodeChunk] = []
    i = 0

    while i < len(lines):
        end = min(i + WINDOW_LINES, len(lines))
        code = "\n".join(lines[i:end])

        if code.strip():
            # Best-effort: check if this chunk starts with a function/method declaration
            fname = _detect_function_name(lines[i], file.language)

            chunks.append(CodeChunk(
                chunk_id="",
                file_path=file.path,
                language=file.language,
                repo=file.repo,
                code=code[:MAX_CHUNK_CHARS],
                start_line=i + 1,
                end_line=end,
                function_name=fname,
            ))

        step = max(1, WINDOW_LINES - OVERLAP_LINES)
        i += step

    return chunks


# ── Helpers ───────────────────────────────────────────────────────────────────
_FUNC_PATTERNS: dict[str, re.Pattern] = {
    "javascript": re.compile(r"(?:async\s+)?function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\("),
    "typescript": re.compile(r"(?:async\s+)?function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\("),
    "java":       re.compile(r"(?:public|private|protected|static|\s)+\w[\w<>[\]]*\s+(\w+)\s*\("),
    "go":         re.compile(r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\("),
    "rust":       re.compile(r"^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)"),
    "cpp":        re.compile(r"(?:\w[\w:*&<> ]+)\s+(\w+)\s*\("),
    "c":          re.compile(r"(?:\w[\w *]+)\s+(\w+)\s*\("),
    "csharp":     re.compile(r"(?:public|private|protected|static|\s)+\w[\w<>[\]]*\s+(\w+)\s*\("),
    "kotlin":     re.compile(r"(?:fun\s+)(\w+)\s*\("),
    "swift":      re.compile(r"(?:func\s+)(\w+)\s*\("),
    "ruby":       re.compile(r"def\s+(\w+)"),
    "php":        re.compile(r"function\s+(\w+)\s*\("),
    "bash":       re.compile(r"^\s*(?:function\s+)?(\w+)\s*\(\s*\)"),
}


def _detect_function_name(line: str, language: str) -> str:
    """Try to extract a function / method name from a single line."""
    pat = _FUNC_PATTERNS.get(language)
    if pat is None:
        return ""
    m = pat.search(line.strip())
    if not m:
        return ""
    # Return first non-None capture group
    return next((g for g in m.groups() if g), "")


def _make_id(chunk: CodeChunk) -> str:
    """Create a stable, URL-safe unique ID for a chunk."""
    # Use a hash of (repo, path, start_line) so re-ingesting the same repo
    # produces the same IDs → enables upsert without duplicates.
    raw = f"{chunk.repo}::{chunk.file_path}::{chunk.start_line}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]
