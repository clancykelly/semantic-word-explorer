#!/usr/bin/env python3
"""Build our own lexicon from the kaikki.org wiktextract English dump.

Produces, under data/lexicon/:
  - lexicon.db        SQLite: words(word, pos, registers, gloss) + links(src, dst, kind)
  - vocab.txt         one word per line (input to --embed)
  - embeddings.npy    (N, D) float32 Model2Vec vectors over the vocabulary
  - word2idx.json     word -> row index

Why: existing thesaurus APIs are frequency-biased and unlabeled. Wiktionary
carries REGISTER tags (archaic, poetic, literary, dialectal, rare, ...) — the
exact signal for surfacing deep-cut vocabulary — and its long tail goes far
beyond what Datamuse returns. Form-of/alt-of/misspelling entries are skipped,
which removes the junk tail ("walke", "walkin") that corpus-frequency rarity
mistakes for rare words.

Usage:
  python build_lexicon.py            # parse dump -> lexicon.db + vocab.txt
  python build_lexicon.py --embed    # vocab.txt -> embeddings.npy + word2idx.json
"""

import gzip
import json
import re
import sqlite3
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "lexicon"
DUMP = DATA_DIR / "kaikki-en.jsonl.gz"
DB = DATA_DIR / "lexicon.db"
VOCAB = DATA_DIR / "vocab.txt"

# Parts of speech worth keeping for a thesaurus
KEEP_POS = {
    "noun", "verb", "adj", "adv", "prep", "conj", "pron", "det", "intj",
    "particle", "postp", "article", "contraction",
}

# Register labels we index (the product's core signal)
REGISTERS = {
    "archaic", "obsolete", "dated", "poetic", "literary", "formal", "informal",
    "colloquial", "slang", "dialectal", "dialect", "rare", "humorous", "vulgar",
    "derogatory", "offensive", "historical", "regional",
}

LINK_KINDS = ("synonyms", "related", "hyponyms", "hypernyms")

WORD_RE = re.compile(r"^[a-z][a-z'-]*[a-z]$")


def valid_word(w: str) -> bool:
    return (
        isinstance(w, str)
        and 2 <= len(w) <= 30
        and w == w.lower()
        and bool(WORD_RE.fullmatch(w))
        and "--" not in w
        and "''" not in w
    )


def parse() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE words (word TEXT PRIMARY KEY, pos TEXT, registers TEXT, gloss TEXT)")
    con.execute("CREATE TABLE links (src TEXT, dst TEXT, kind TEXT)")

    words: dict[str, dict] = {}
    link_batch: list[tuple[str, str, str]] = []
    n_lines = n_kept_senses = 0

    def flush_links():
        nonlocal link_batch
        if link_batch:
            con.executemany("INSERT INTO links VALUES (?,?,?)", link_batch)
            link_batch = []

    with gzip.open(DUMP, "rt", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            if n_lines % 200_000 == 0:
                print(f"  {n_lines} lines · {len(words)} words kept", flush=True)
                flush_links()
                con.commit()
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue

            word = e.get("word", "")
            pos = e.get("pos", "")
            if pos not in KEEP_POS or not valid_word(word):
                continue

            regs: set[str] = set()
            gloss = None
            kept_any = False
            for sense in e.get("senses", []):
                tags = set(sense.get("tags") or [])
                # Skip inflections, alternative/obsolete spellings, misspellings —
                # the junk tail that masquerades as "rare".
                if (
                    sense.get("form_of")
                    or sense.get("alt_of")
                    or "misspelling" in tags
                    or "alt-of" in tags
                    or "form-of" in tags
                    or "abbreviation" in tags
                    or "initialism" in tags
                ):
                    continue
                # Register labels from the PRIMARY sense only: a word-level
                # union would mark common words as archaic because one obscure
                # sense is ("pathetic", "pity").
                if not kept_any:
                    regs |= tags & REGISTERS
                kept_any = True
                if gloss is None:
                    gl = sense.get("glosses") or []
                    if gl:
                        gloss = gl[0][:220]
                # Sense-level links
                for kind in LINK_KINDS:
                    for item in sense.get(kind) or []:
                        dst = (item.get("word") or "").lower()
                        if valid_word(dst) and dst != word:
                            link_batch.append((word, dst, kind))

            if not kept_any:
                continue
            n_kept_senses += 1

            # Entry-level links
            for kind in LINK_KINDS:
                for item in e.get(kind) or []:
                    dst = (item.get("word") or "").lower()
                    if valid_word(dst) and dst != word:
                        link_batch.append((word, dst, kind))

            rec = words.setdefault(word, {"pos": set(), "regs": set(), "gloss": None})
            rec["pos"].add(pos)
            rec["regs"] |= regs
            if rec["gloss"] is None:
                rec["gloss"] = gloss

    flush_links()
    con.executemany(
        "INSERT OR REPLACE INTO words VALUES (?,?,?,?)",
        (
            (w, ",".join(sorted(r["pos"])), ",".join(sorted(r["regs"])), r["gloss"])
            for w, r in words.items()
        ),
    )
    con.execute("CREATE INDEX idx_links_src ON links(src)")
    con.execute("CREATE INDEX idx_links_dst ON links(dst)")
    con.commit()

    n_links = con.execute("SELECT COUNT(*) FROM links").fetchone()[0]
    n_reg = con.execute("SELECT COUNT(*) FROM words WHERE registers != ''").fetchone()[0]
    con.close()

    VOCAB.write_text("\n".join(sorted(words)) + "\n", encoding="utf-8")
    print(f"DONE parse: {n_lines} lines -> {len(words)} words ({n_reg} register-tagged), {n_links} links")


def embed() -> None:
    import numpy as np

    sys.path.insert(0, str(Path(__file__).parent))
    from app.embeddings import load_word_vectors

    vocab = VOCAB.read_text(encoding="utf-8").split()
    vectors = load_word_vectors()
    if vectors is None:
        raise SystemExit("no word vectors available")
    print(f"embedding {len(vocab)} words with {vectors.model_name}…", flush=True)

    out = np.zeros((len(vocab), vectors.dim), dtype=np.float32)
    B = 20_000
    for i in range(0, len(vocab), B):
        out[i : i + B] = vectors.encode(vocab[i : i + B])
        print(f"  {min(i + B, len(vocab))}/{len(vocab)}", flush=True)

    np.save(DATA_DIR / "embeddings.npy", out)
    (DATA_DIR / "word2idx.json").write_text(
        json.dumps({w: i for i, w in enumerate(vocab)}), encoding="utf-8"
    )
    print(f"DONE embed: {out.shape} -> {DATA_DIR/'embeddings.npy'}")


if __name__ == "__main__":
    if "--embed" in sys.argv:
        embed()
    else:
        parse()
