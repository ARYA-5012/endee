"""
endee_client.py
───────────────
A clean wrapper around the Endee Python SDK that handles all vector-database
operations for DevSearch:

  • create_index      — one index per GitHub repo
  • upsert_chunks     — store embedded code chunks with metadata
  • search            — semantic nearest-neighbour search
  • list_indexes      — show all indexed repos
  • delete_index      — remove a repo from the database
  • index_exists      — check before re-creating
  • ping              — verify Endee is reachable

Endee stores each vector alongside an arbitrary `meta` dict.
We exploit this to make search results completely self-contained:
the code text, file path, language, and line numbers all live in `meta`
so we never need a secondary lookup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.chunker import CodeChunk

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_BASE_URL = "http://localhost:8080/api/v1"
EMBEDDING_DIM    = 384
UPSERT_BATCH     = 128     # vectors per upsert call — keep memory usage low

# Separator for index names. Using triple underscore because GitHub
# usernames and repo names cannot contain underscores consecutively,
# making this collision-safe (unlike "--" which can appear in repo names).
INDEX_SEP = "___"


# ── Result model ──────────────────────────────────────────────────────────────
@dataclass
class SearchResult:
    chunk_id:      str
    similarity:    float
    file_path:     str
    language:      str
    repo:          str
    function_name: str
    start_line:    int
    end_line:      int
    code:          str


# ── Client ────────────────────────────────────────────────────────────────────
class DevSearchDB:
    """Thin, DevSearch-flavoured wrapper around the Endee SDK."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        auth_token: Optional[str] = None,
    ) -> None:
        from endee import Endee, Precision  # imported here so module loads even if SDK missing

        self._Precision = Precision

        url   = (base_url or os.getenv("ENDEE_BASE_URL", DEFAULT_BASE_URL)).strip()
        token = (auth_token or os.getenv("ENDEE_AUTH_TOKEN", "") or "").strip()

        self._client = Endee(token) if token else Endee()
        self._client.set_base_url(url)

    # ── Connection check ─────────────────────────────────────────────────────
    def ping(self) -> bool:
        """Return True if Endee is reachable. Raises on failure."""
        # list_indexes is the lightest authenticated call available
        self._client.list_indexes()
        return True

    # ── Index lifecycle ───────────────────────────────────────────────────────
    def index_name_for_repo(self, repo: str) -> str:
        """Convert 'owner/repo' to a valid Endee index name.
        
        Endee index names must be alphanumeric + underscores only.
        We use triple-underscore as the owner/repo separator,
        and replace hyphens with double-underscores.
        """
        return repo.lower().replace("/", INDEX_SEP).replace("-", "__")

    def repo_name_from_index(self, index_name: str) -> str:
        """Convert an Endee index name back to 'owner/repo'."""
        # Restore hyphens first (double-underscore → hyphen),
        # then restore the slash (triple-underscore → slash)
        name = index_name.replace(INDEX_SEP, "/", 1)
        name = name.replace("__", "-")
        return name

    def index_exists(self, repo: str) -> bool:
        try:
            indexes = self._client.list_indexes()
            name = self.index_name_for_repo(repo)
            names = [
                (i.name if hasattr(i, "name") else i.get("name", ""))
                for i in (indexes or [])
            ]
            return name in names
        except Exception:
            return False

    def create_index(self, repo: str, skip_exists_check: bool = False) -> None:
        """Create a new Endee index for the given repo.

        Parameters
        ----------
        skip_exists_check : if True, skip the index_exists() call (caller
                            already verified). Avoids redundant API calls.
        """
        name = self.index_name_for_repo(repo)
        if skip_exists_check or not self.index_exists(repo):
            self._client.create_index(
                name=name,
                dimension=EMBEDDING_DIM,
                space_type="cosine",
                precision="float32",
            )

    def delete_index(self, repo: str) -> bool:
        """Delete an index. Returns True on success, False on failure."""
        name = self.index_name_for_repo(repo)
        try:
            self._client.delete_index(name)
            return True
        except Exception as e:
            import logging
            logging.warning(f"Failed to delete index '{name}': {e}")
            return False

    def list_indexes(self) -> List[Dict[str, Any]]:
        """Return a list of index info dicts.

        NOTE: This method intentionally lets exceptions propagate so callers
        can detect when Endee is unreachable.
        """
        indexes = self._client.list_indexes() or []
        result = []
        for i in indexes:
            if hasattr(i, "__dict__"):
                result.append(vars(i))
            elif isinstance(i, dict):
                result.append(i)
        return result

    # ── Upserting vectors ─────────────────────────────────────────────────────
    def upsert_chunks(
        self,
        repo:       str,
        chunks:     List[CodeChunk],
        embeddings: List[List[float]],
    ) -> int:
        """
        Store code chunks + their embeddings in Endee.

        Returns the number of vectors successfully upserted.
        """
        index = self._client.get_index(self.index_name_for_repo(repo))
        upserted = 0

        # Process in batches to keep memory under control
        for start in range(0, len(chunks), UPSERT_BATCH):
            batch_chunks = chunks[start : start + UPSERT_BATCH]
            batch_embeds = embeddings[start : start + UPSERT_BATCH]

            items = []
            for chunk, vector in zip(batch_chunks, batch_embeds):
                items.append({
                    "id":     chunk.chunk_id,
                    "vector": vector,
                    "meta": {
                        "file_path":     chunk.file_path,
                        "language":      chunk.language,
                        "repo":          chunk.repo,
                        "function_name": chunk.function_name,
                        "start_line":    chunk.start_line,
                        "end_line":      chunk.end_line,
                        # Store truncated code in meta for self-contained results
                        "code": chunk.code[:2000],
                    },
                })

            index.upsert(items)
            upserted += len(items)

        return upserted

    # ── Querying ──────────────────────────────────────────────────────────────
    def search(
        self,
        repo:        str,
        query_vector: List[float],
        top_k:       int = 8,
        language:    Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Run a nearest-neighbour query against the repo's Endee index.

        Parameters
        ----------
        repo         : "owner/reponame"
        query_vector : 384-dim float list produced by embedder.embed_single()
        top_k        : number of results to return
        language     : optional language filter (post-filter on results)
        """
        try:
            index = self._client.get_index(self.index_name_for_repo(repo))
            # Only over-fetch when we need to post-filter by language
            fetch_k = top_k * 2 if language else top_k
            raw = index.query(vector=query_vector, top_k=fetch_k)
        except Exception as e:
            raise RuntimeError(f"Endee query failed: {e}") from e

        results: List[SearchResult] = []
        for r in (raw or []):
            meta = r.meta if hasattr(r, "meta") else r.get("meta", {})
            sim  = r.similarity if hasattr(r, "similarity") else r.get("similarity", 0.0)
            rid  = r.id if hasattr(r, "id") else r.get("id", "")

            # Post-filter by language if requested
            if language and meta.get("language") != language:
                continue

            results.append(SearchResult(
                chunk_id      = rid,
                similarity    = round(float(sim), 4),
                file_path     = meta.get("file_path", ""),
                language      = meta.get("language", ""),
                repo          = meta.get("repo", repo),
                function_name = meta.get("function_name", ""),
                start_line    = int(meta.get("start_line", 0)),
                end_line      = int(meta.get("end_line", 0)),
                code          = meta.get("code", ""),
            ))

            if len(results) >= top_k:
                break

        return results
