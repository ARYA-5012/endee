"""
embedder.py
───────────
Wraps sentence-transformers to produce 384-dimensional embeddings for code
chunks and natural-language queries.

Model choice: all-MiniLM-L6-v2
• Fast (< 50 ms per batch on CPU)
• 384 dimensions — low storage overhead in Endee
• Strong semantic quality for both code and natural language
• Freely downloadable from HuggingFace Hub (~90 MB)

The model is loaded once and cached in the module-level singleton so
Streamlit reruns don't reload it from disk on every interaction.
"""

from __future__ import annotations

from typing import List

import numpy as np

# Lazy import — only load the heavy library when first needed
_model = None
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def _get_model():
    """Return the singleton SentenceTransformer instance (load on first call)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """
    Embed a list of strings and return a list of float32 vectors.

    Parameters
    ----------
    texts:      list of raw strings (code snippets or natural-language queries)
    batch_size: number of texts processed per forward pass (tune for your RAM)

    Returns
    -------
    List of 384-dimensional float lists, one per input text.
    """
    model = _get_model()
    # SentenceTransformer returns a numpy array; convert to plain Python lists
    # because Endee's SDK expects Python float lists.
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,   # cosine similarity = dot product after L2 norm
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def embed_single(text: str) -> List[float]:
    """Convenience wrapper for embedding a single string (e.g. a search query)."""
    return embed_texts([text])[0]
