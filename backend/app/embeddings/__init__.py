"""Embedding service module.

The app has a single search provider — Datamuse-backed semantic search. Static
word vectors (Model2Vec, loaded via ``load_word_vectors``) are fed into it for
relevance filtering and meaning-based clustering; they are not a separate mode.
"""

import os

from .base import EmbeddingProvider
from .datamuse_provider import DatamuseProvider

__all__ = [
    "EmbeddingProvider",
    "DatamuseProvider",
    "get_datamuse_provider",
    "load_word_vectors",
]


def get_datamuse_provider() -> DatamuseProvider:
    """Get the Datamuse provider for semantic similarity."""
    return DatamuseProvider()


def load_word_vectors(model_name: str | None = None):
    """Load static word vectors for relevance + clustering.

    Prefers Model2Vec (best quality, encodes any word) but ONLY if the model is
    already cached locally — the load is forced offline so server startup never
    hangs on a slow/blocked download. Falls back to an on-disk vector table
    (``EMBEDDING_DATA_DIR``, default ``data/glove-6b-300d``). Returns ``None`` if
    neither is available (clustering then falls back to part-of-speech grouping).

    A background ``model2vec`` download populates the cache; a later restart then
    picks up Model2Vec automatically.
    """
    name = model_name or os.getenv("EMBEDDING_MODEL")

    # 1. Model2Vec, cache-only — never trigger or await a network download here.
    prev_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from .vectors import DEFAULT_MODEL, Model2VecVectors

        return Model2VecVectors(name or DEFAULT_MODEL)
    except Exception as e:  # noqa: BLE001 - not cached / unavailable; fall back
        print(f"Model2Vec not loaded from cache ({type(e).__name__}); trying on-disk table")
    finally:
        if prev_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_offline

    # 2. On-disk vector table fallback.
    table_dir = os.getenv("EMBEDDING_DATA_DIR", "data/glove-6b-300d")
    try:
        from pathlib import Path

        from .vectors import TableVectors

        if (Path(table_dir) / "embeddings.npy").exists():
            return TableVectors(table_dir)
        print(f"No vector table at {table_dir}")
    except Exception as e:  # noqa: BLE001
        print(f"Vector table unavailable ({e})")

    print("No word vectors available; clustering will fall back to POS grouping")
    return None
