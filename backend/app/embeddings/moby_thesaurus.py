"""Moby Thesaurus II loader and query interface.

Moby Thesaurus II is a public domain thesaurus with ~30,000 root words
and over 2.5 million synonyms/related words. It's particularly good for
literary and poetic synonyms that modern APIs like Datamuse may miss.

Data source: Project Gutenberg
https://www.gutenberg.org/files/3202/files/mthesaur.txt
"""

from pathlib import Path


class MobyThesaurus:
    """Query interface for Moby Thesaurus II."""

    def __init__(self, data_path: str | Path | None = None):
        """Initialize the thesaurus.

        Args:
            data_path: Path to mthesaur.txt file. If None, uses default location.
        """
        if data_path is None:
            # Default to data directory relative to this file
            data_path = Path(__file__).parent.parent.parent / "data" / "mthesaur.txt"
        else:
            data_path = Path(data_path)

        self._entries: dict[str, list[str]] = {}
        self._reverse_index: dict[str, set[str]] = {}  # word -> headwords containing it
        self._loaded = False
        self._data_path = data_path

    def _load(self) -> None:
        """Load the thesaurus data lazily."""
        if self._loaded:
            return

        if not self._data_path.exists():
            print(f"Moby Thesaurus not found at {self._data_path}")
            self._loaded = True
            return

        print(f"Loading Moby Thesaurus from {self._data_path}...")

        with open(self._data_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) < 2:
                    continue

                headword = parts[0].lower().strip()
                synonyms = [p.strip() for p in parts[1:] if p.strip()]

                self._entries[headword] = synonyms

                # Build reverse index for bidirectional lookup
                for syn in synonyms:
                    syn_lower = syn.lower()
                    if syn_lower not in self._reverse_index:
                        self._reverse_index[syn_lower] = set()
                    self._reverse_index[syn_lower].add(headword)

        print(f"Loaded {len(self._entries)} Moby Thesaurus entries")
        self._loaded = True

    def get_synonyms(self, word: str, include_reverse: bool = True) -> list[str]:
        """Get synonyms for a word.

        Args:
            word: The word to look up
            include_reverse: Also include entries where this word appears as a synonym

        Returns:
            List of synonyms/related words (may include phrases)
        """
        self._load()

        word_lower = word.lower().strip()
        results: set[str] = set()

        # Direct lookup
        if word_lower in self._entries:
            results.update(self._entries[word_lower])

        # Reverse lookup - find entries where this word is a synonym
        if include_reverse and word_lower in self._reverse_index:
            for headword in self._reverse_index[word_lower]:
                # Add the headword itself
                results.add(headword)
                # Optionally add siblings (other synonyms of the headword)
                # This can add a lot of words, so we limit it
                # results.update(self._entries.get(headword, [])[:20])

        # Remove the query word itself
        results.discard(word_lower)

        return list(results)

    def get_phrases(self, word: str) -> list[str]:
        """Get only multi-word phrase synonyms for a word.

        Args:
            word: The word to look up

        Returns:
            List of phrase synonyms (words containing spaces)
        """
        all_synonyms = self.get_synonyms(word)
        return [s for s in all_synonyms if " " in s]

    def has_word(self, word: str) -> bool:
        """Check if a word exists in the thesaurus."""
        self._load()
        word_lower = word.lower().strip()
        return word_lower in self._entries or word_lower in self._reverse_index

    @property
    def entry_count(self) -> int:
        """Number of headword entries."""
        self._load()
        return len(self._entries)


# Singleton instance
_moby_thesaurus: MobyThesaurus | None = None


def get_moby_thesaurus() -> MobyThesaurus:
    """Get the singleton Moby Thesaurus instance."""
    global _moby_thesaurus
    if _moby_thesaurus is None:
        _moby_thesaurus = MobyThesaurus()
    return _moby_thesaurus
