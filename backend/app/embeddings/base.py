"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class WordResult:
    """Result from embedding lookup."""

    word: str
    similarity: float
    coordinates: tuple[float, float]
    frequency: str  # "common", "uncommon", "rare"
    cluster: int
    formality: float  # 0.0 = casual, 1.0 = formal


@dataclass
class SenseInfo:
    """Information about a word sense."""

    sense: str  # e.g., "bank|NOUN:financial"
    label: str  # e.g., "bank (financial institution)"
    frequency: int  # relative frequency


@dataclass
class SearchResult:
    """Full search result including neighbors and metadata."""

    word: str
    normalized_word: str
    sense: str | None
    available_senses: list[SenseInfo]
    neighbors: list[WordResult]
    clusters: list[dict]


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    Implementations can use mock data, FAISS, or other backends.
    """

    @abstractmethod
    def search(
        self,
        word: str,
        sense: str | None = None,
        limit: int = 100,
    ) -> SearchResult | None:
        """Search for semantically related words.

        Args:
            word: The word to search for (will be normalized)
            sense: Optional specific sense to use (e.g., "bank|NOUN:financial")
            limit: Maximum number of neighbors to return

        Returns:
            SearchResult if word found, None if not found
        """
        pass

    @abstractmethod
    def has_word(self, word: str) -> bool:
        """Check if a word exists in the vocabulary."""
        pass

    @abstractmethod
    def find_similar_word(self, word: str) -> str | None:
        """Find a similar word for typo correction.

        Returns the closest word within edit distance 2, or None.
        """
        pass

    @abstractmethod
    def get_suggestions(self, limit: int = 3) -> list[str]:
        """Get suggestion words when a word is not found."""
        pass

    @abstractmethod
    def get_available_words(self) -> list[str]:
        """Get list of all available words in vocabulary."""
        pass

    def expand_cluster(
        self,
        word: str,
        anchor_words: list[str],
        limit: int = 50,
        exclude_words: list[str] | None = None,
    ) -> SearchResult | None:
        """Expand a semantic cluster by fetching more words similar to anchor words.

        Args:
            word: The original query word
            anchor_words: Representative words from the cluster to expand
            limit: Number of additional words to fetch
            exclude_words: Words already shown (to avoid duplicates)

        Returns:
            SearchResult with additional words, or None if not supported
        """
        # Default implementation returns None (not supported)
        return None
