"""Our own lexicon: Wiktionary-derived words, register labels, per-sense
glosses, and a private semantic index.

Built by ``build_lexicon.py`` from the kaikki.org wiktextract dump. Provides:
  - links(word):    curated synonym/related/hypo/hypernym edges
  - knn(word):      nearest neighbors in OUR gloss-aware embedding index
  - glosses(words): every kept sense's definition — the basis of gloss-aware
                    relevance scoring (a static model is near-random on rare
                    single tokens; a definition carries the meaning)
  - registers():    archaic/poetic/literary/... labels, the deep-cut signal

Thread-safe for FastAPI's threadpool (SQLite guarded by a lock; the embedding
matrix is a read-only memmap).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import numpy as np

from .vectors import WordVectors

DEEP_REGISTERS = frozenset(
    {"archaic", "obsolete", "poetic", "literary", "rare", "dated", "dialectal", "dialect"}
)

_DATA_DIR = Path(__file__).parents[2] / "data" / "lexicon"


class Lexicon:
    def __init__(self, data_dir: Path, vectors: WordVectors):
        self._con = sqlite3.connect(data_dir / "lexicon.db", check_same_thread=False)
        self._lock = threading.Lock()
        self._vectors = vectors
        self._emb = np.load(data_dir / "embeddings.npy", mmap_mode="r")
        with open(data_dir / "word2idx.json") as f:
            self._w2i: dict[str, int] = json.load(f)
        self._idx2w = [""] * len(self._w2i)
        for w, i in self._w2i.items():
            self._idx2w[i] = w
        self._has_glosses = bool(
            self._con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='glosses'"
            ).fetchone()
        )
        print(
            f"Lexicon loaded: {len(self._w2i)} words, index {self._emb.shape}, "
            f"per-sense glosses: {'yes' if self._has_glosses else 'NO (rebuild)'}"
        )

    def links(self, word: str, cap: int = 200) -> list[str]:
        """Curated outgoing + incoming edges for a word."""
        with self._lock:
            rows = self._con.execute(
                "SELECT dst FROM links WHERE src = ? UNION SELECT src FROM links WHERE dst = ? LIMIT ?",
                (word, word, cap),
            ).fetchall()
        return [r[0] for r in rows if r[0] != word]

    def glosses(self, words: list[str], per_word: int = 3) -> dict[str, list[str]]:
        """Batch: word -> up to ``per_word`` sense glosses (single words only)."""
        out: dict[str, list[str]] = {}
        if not self._has_glosses:
            return out
        uniq = list(dict.fromkeys(w.lower() for w in words if " " not in w))
        with self._lock:
            for i in range(0, len(uniq), 500):
                chunk = uniq[i : i + 500]
                q = ",".join("?" * len(chunk))
                for w, g in self._con.execute(
                    f"SELECT word, gloss FROM glosses WHERE word IN ({q}) AND rank < ? ORDER BY word, rank",
                    (*chunk, per_word),
                ):
                    out.setdefault(w, []).append(g)
        return out

    def text_for(self, word: str) -> str:
        """The gloss-aware text a word is embedded as ("word: gloss; gloss")."""
        gl = self.glosses([word]).get(word.lower(), [])
        return f"{word}: {'; '.join(g[:120] for g in gl[:3])}" if gl else word

    def knn(self, word: str, k: int = 120) -> list[tuple[str, float]]:
        """Nearest neighbors of a word in our own (gloss-aware) index."""
        qv = self._vectors.encode([self.text_for(word.lower())])[0]
        sims = self._emb @ qv
        k = min(k + 1, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        out = []
        for i in top:
            w = self._idx2w[int(i)]
            if w != word.lower():
                out.append((w, float(sims[int(i)])))
        return out[:k]

    def registers(self, words: list[str]) -> dict[str, set[str]]:
        """Batch register lookup: word -> {archaic, poetic, ...} (empty if none)."""
        out: dict[str, set[str]] = {}
        uniq = list(dict.fromkeys(w.lower() for w in words))
        with self._lock:
            for i in range(0, len(uniq), 500):
                chunk = uniq[i : i + 500]
                q = ",".join("?" * len(chunk))
                for w, regs in self._con.execute(
                    f"SELECT word, registers FROM words WHERE word IN ({q}) AND registers != ''",
                    chunk,
                ):
                    out[w] = set(regs.split(","))
        return out

    def candidates(self, word: str, k: int = 120) -> dict[str, dict]:
        """Candidate words for a query: curated links + our-index neighbors.

        Returns {word: {"registers": set, "link": bool}} — ``link`` marks words
        backed by a curated Wiktionary edge (evidence), as opposed to pure
        vector neighbors.
        """
        link_words = self.links(word)
        nn = [w for w, _ in self.knn(word, k)]
        pool = list(dict.fromkeys(link_words + nn))
        regs = self.registers(pool)
        linkset = set(link_words)
        return {w: {"registers": regs.get(w, set()), "link": w in linkset} for w in pool}


def load_lexicon(vectors: WordVectors, data_dir: Path | None = None) -> Lexicon | None:
    """Load the lexicon if its build artifacts exist; None otherwise."""
    d = data_dir or _DATA_DIR
    needed = ["lexicon.db", "embeddings.npy", "word2idx.json"]
    if not all((d / n).exists() for n in needed):
        print(f"Lexicon not built (missing artifacts in {d}) — run build_lexicon.py")
        return None
    try:
        return Lexicon(d, vectors)
    except Exception as e:  # noqa: BLE001 - degrade gracefully
        print(f"Lexicon unavailable ({e})")
        return None
