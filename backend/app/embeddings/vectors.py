"""Static word vectors for relevance scoring + clustering.

Two interchangeable backends behind one interface:

- ``Model2VecVectors`` (preferred): a small distilled static model that encodes
  ANY word on demand, including rare/archaic ones (subword tokenizer). ~30MB,
  cached by Hugging Face after first download.
- ``TableVectors`` (fallback): a precomputed vector table on disk
  (``embeddings.npy`` + ``word2idx.json``). Reliable and offline; only covers
  its fixed vocabulary (out-of-vocab words get a zero vector).

Either way, relevance scoring and the clustering distance matrix are single
vectorized numpy matmuls.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# potion-base-8M: distilled from BGE-base, ~30MB, strong single-word quality.
DEFAULT_MODEL = "minishlab/potion-base-8M"


class WordVectors:
    """Base interface: encode words to L2-normalized vectors, with caching."""

    model_name: str = "base"
    dim: int = 0

    def __init__(self) -> None:
        self._cache: dict[str, np.ndarray] = {}

    def _encode_uncached(self, words: list[str]) -> np.ndarray:
        """Return raw (un-normalized) (N, D) vectors for the given words."""
        raise NotImplementedError

    def encode(self, words: list[str]) -> np.ndarray:
        """Return an (N, D) array of L2-normalized vectors for ``words``."""
        if not words:
            return np.zeros((0, self.dim), dtype=np.float32)

        missing = [w for w in dict.fromkeys(words) if w not in self._cache]
        if missing:
            vecs = np.asarray(self._encode_uncached(missing), dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
            for w, v in zip(missing, vecs):
                self._cache[w] = v

        return np.stack([self._cache[w] for w in words])

    def encode_one(self, word: str) -> np.ndarray:
        """Return the L2-normalized vector for a single word."""
        return self.encode([word])[0]

    def similarities(self, query: str, words: list[str]) -> np.ndarray:
        """Cosine similarity of each word to ``query`` (vectorized)."""
        if not words:
            return np.zeros(0, dtype=np.float32)
        q = self.encode_one(query)
        matrix = self.encode(words)
        return (matrix @ q).astype(np.float32)

    def distance_matrix(self, words: list[str]) -> np.ndarray:
        """Pairwise cosine-distance matrix (1 - cosine) for ``words``."""
        if not words:
            return np.zeros((0, 0), dtype=np.float32)
        matrix = self.encode(words)
        dist = 1.0 - (matrix @ matrix.T)
        np.clip(dist, 0.0, 2.0, out=dist)
        np.fill_diagonal(dist, 0.0)
        return dist.astype(np.float32)


class Model2VecVectors(WordVectors):
    """Preferred backend: distilled static model, encodes any word on demand."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        super().__init__()
        from model2vec import StaticModel

        self.model_name = model_name
        self._model = StaticModel.from_pretrained(model_name)
        self.dim = int(np.asarray(self._model.encode(["test"])).shape[1])

    def _encode_uncached(self, words: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(words), dtype=np.float32)


class TableVectors(WordVectors):
    """Fallback backend: precomputed static vectors from a table on disk.

    Out-of-vocabulary single words get a zero vector (cosine 0). Phrases are
    averaged over the component words present in the table.
    """

    def __init__(self, data_dir: str | Path):
        super().__init__()
        path = Path(data_dir)
        self.model_name = f"table:{path.name}"
        self._emb = np.load(path / "embeddings.npy", mmap_mode="r")
        with open(path / "word2idx.json") as f:
            self._w2i: dict[str, int] = json.load(f)
        self.dim = int(self._emb.shape[1])

    def _encode_uncached(self, words: list[str]) -> np.ndarray:
        out = np.zeros((len(words), self.dim), dtype=np.float32)
        for i, word in enumerate(words):
            wl = word.lower()
            idx = self._w2i.get(wl)
            if idx is not None:
                out[i] = self._emb[idx]
            elif " " in wl:
                parts = [self._w2i[p] for p in wl.split() if p in self._w2i]
                if parts:
                    out[i] = np.asarray(self._emb[parts], dtype=np.float32).mean(axis=0)
            # else: out-of-vocabulary -> leave as zero vector
        return out
