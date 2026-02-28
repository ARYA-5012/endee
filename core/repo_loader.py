"""
repo_loader.py
──────────────
Fetches all code files from a public GitHub repository using the
GitHub REST API (no git clone required — just HTTP requests).

Supports up to 200 files per repo and respects the 100 KB per-file limit.
Provide a GITHUB_TOKEN in .env to raise the rate limit from 60 → 5 000 req/hr.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import requests

logger = logging.getLogger(__name__)

# ── Supported languages & their file extensions ──────────────────────────────
EXTENSION_MAP: dict[str, str] = {
    ".py":   "python",
    ".js":   "javascript",
    ".ts":   "typescript",
    ".jsx":  "javascript",
    ".tsx":  "typescript",
    ".java": "java",
    ".go":   "go",
    ".cpp":  "cpp",
    ".cc":   "cpp",
    ".cxx":  "cpp",
    ".c":    "c",
    ".rs":   "rust",
    ".rb":   "ruby",
    ".php":  "php",
    ".swift":"swift",
    ".kt":   "kotlin",
    ".cs":   "csharp",
    ".sh":   "bash",
}

MAX_FILE_BYTES   = 100_000   # skip files larger than 100 KB
MAX_FILES        = 200       # max files to fetch per repo (API rate-limit safety)
REQUEST_TIMEOUT  = 15        # seconds


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class CodeFile:
    path:     str           # relative path inside the repo  e.g. "src/auth/jwt.py"
    language: str           # e.g. "python"
    content:  str           # raw source code
    repo:     str           # "owner/reponame"
    size:     int = 0       # bytes


# ── URL parsing ───────────────────────────────────────────────────────────────
def parse_github_url(url: str) -> tuple[str, str]:
    """Return (owner, repo) from any GitHub URL variant."""
    url = url.strip().rstrip("/")
    patterns = [
        r"github\.com[:/]([^/]+)/([^/\s]+?)(?:\.git)?$",
        r"github\.com/([^/]+)/([^/\s]+)",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1), m.group(2).replace(".git", "")
    raise ValueError(f"Cannot parse GitHub URL: {url!r}")


# ── Main fetcher ──────────────────────────────────────────────────────────────
def fetch_repo_files(
    github_url: str,
    github_token: Optional[str] = None,
) -> Generator[tuple[CodeFile, int, int], None, None]:
    """
    Yields (CodeFile, current_index, total_count) tuples so callers can
    display a live progress bar while files are being fetched.

    Usage::

        for file, idx, total in fetch_repo_files(url, token):
            progress = idx / total
    """
    owner, repo = parse_github_url(github_url)

    headers: dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    # 1. Discover default branch ──────────────────────────────────────────────
    r = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}",
        headers=headers, timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    default_branch = r.json().get("default_branch", "main")

    # 2. Get full file tree (recursive) ───────────────────────────────────────
    r = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1",
        headers=headers, timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    tree_data = r.json()

    # Warn if the tree was truncated by GitHub (>100k entries)
    if tree_data.get("truncated"):
        logger.warning(
            "GitHub tree for %s/%s was truncated — some files may be missing "
            "from the index. Consider narrowing the repo scope.",
            owner, repo,
        )

    # 3. Filter to code files within size budget ───────────────────────────────
    blobs: list[dict] = [
        item for item in tree_data.get("tree", [])
        if item["type"] == "blob"
        and Path(item["path"]).suffix.lower() in EXTENSION_MAP
        and item.get("size", 0) < MAX_FILE_BYTES
        and item.get("size", 0) > 10          # skip near-empty files
        and not _is_vendored(item["path"])     # skip vendor / node_modules
    ][:MAX_FILES]

    total = len(blobs)
    if total == 0:
        return

    base_raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{default_branch}"

    # 4. Fetch each file ──────────────────────────────────────────────────────
    for idx, item in enumerate(blobs, start=1):
        try:
            resp = requests.get(
                f"{base_raw}/{item['path']}",
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                continue
            content = resp.text
            if not content.strip():
                continue

            lang = EXTENSION_MAP[Path(item["path"]).suffix.lower()]
            yield (
                CodeFile(
                    path=item["path"],
                    language=lang,
                    content=content,
                    repo=f"{owner}/{repo}",
                    size=item.get("size", len(content)),
                ),
                idx,
                total,
            )
        except Exception:
            continue  # skip any file that fails — don't crash the whole ingest


# ── Helpers ───────────────────────────────────────────────────────────────────
_VENDOR_PARTS = {
    "node_modules", "vendor", "third_party", "third-party",
    ".git", "dist", "build", "__pycache__", ".venv", "venv", "env",
    "bower_components", "Pods",
}

def _is_vendored(path: str) -> bool:
    """Return True if the path is inside a known vendored/generated directory."""
    parts = set(Path(path).parts)
    return bool(parts & _VENDOR_PARTS)
