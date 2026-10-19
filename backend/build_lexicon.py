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
    # Every kept sense's gloss (up to 4 per word): relevance scoring compares
    # a candidate's best-matching SENSE to the query, so "lumber" can match
    # "walk" via "to move heavily" even though its first sense is timber.
    con.execute("CREATE TABLE glosses (word TEXT, rank INTEGER, gloss TEXT)")

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

            reg_counts: dict[str, int] = {}
            n_kept_this_entry = 0
            gloss = None
            entry_glosses: list[str] = []
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
                # Count register tags per kept sense; a word is marked only when
                # a MAJORITY of its senses carry the tag. Union over senses marks
                # "pathetic" archaic for one obscure sense; primary-sense-only
                # marks "much" obsolete because Wiktionary lists senses
                # historically. Majority gets both right.
                for t in tags & REGISTERS:
                    reg_counts[t] = reg_counts.get(t, 0) + 1
                n_kept_this_entry += 1
                kept_any = True
                gl = sense.get("glosses") or []
                if gl:
                    g = gl[0][:220]
                    if gloss is None:
                        gloss = g
                    if len(entry_glosses) < 4 and g not in entry_glosses:
                        entry_glosses.append(g)
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

            rec = words.setdefault(
                word, {"pos": set(), "reg_counts": {}, "n_senses": 0, "gloss": None, "glosses": []}
            )
            rec["pos"].add(pos)
            rec["n_senses"] += n_kept_this_entry
            for t, c in reg_counts.items():
                rec["reg_counts"][t] = rec["reg_counts"].get(t, 0) + c
            if rec["gloss"] is None:
                rec["gloss"] = gloss
            for g in entry_glosses:
                if len(rec["glosses"]) >= 4:
                    break
                if g not in rec["glosses"]:
                    rec["glosses"].append(g)

    flush_links()

    def majority_regs(r: dict) -> str:
        n = max(1, r["n_senses"])
        return ",".join(sorted(t for t, c in r["reg_counts"].items() if 2 * c >= n))

    con.executemany(
        "INSERT OR REPLACE INTO words VALUES (?,?,?,?)",
        ((w, ",".join(sorted(r["pos"])), majority_regs(r), r["gloss"]) for w, r in words.items()),
    )
    con.executemany(
        "INSERT INTO glosses VALUES (?,?,?)",
        ((w, i, g) for w, r in words.items() for i, g in enumerate(r["glosses"])),
    )
    con.execute("CREATE INDEX idx_links_src ON links(src)")
    con.execute("CREATE INDEX idx_links_dst ON links(dst)")
    con.execute("CREATE INDEX idx_glosses_word ON glosses(word)")
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

    # Embed "word: gloss; gloss; gloss" rather than the bare token: a static
    # model knows frequent words but is near-random on rare single tokens
    # ("glum" vs "sad" = -0.04), while a definition carries the meaning.
    con = sqlite3.connect(DB)
    gmap: dict[str, list[str]] = {}
    for w, g in con.execute("SELECT word, gloss FROM glosses ORDER BY word, rank"):
        gmap.setdefault(w, []).append(g)
    con.close()
    texts = [
        f"{w}: {'; '.join(g[:120] for g in gmap[w][:3])}" if w in gmap else w
        for w in vocab
    ]
    print(f"embedding {len(vocab)} words ({len(gmap)} with glosses) with {vectors.model_name}…", flush=True)

    out = np.zeros((len(vocab), vectors.dim), dtype=np.float32)
    B = 10_000
    for i in range(0, len(texts), B):
        raw = np.asarray(vectors._encode_uncached(texts[i : i + B]), dtype=np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        out[i : i + B] = raw / norms
        print(f"  {min(i + B, len(texts))}/{len(texts)}", flush=True)

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
