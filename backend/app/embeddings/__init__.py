"""Embedding service module."""

from .base import EmbeddingProvider
from .mock import MockEmbeddingProvider
from .datamuse_provider import DatamuseProvider

__all__ = [
    "EmbeddingProvider",
    "MockEmbeddingProvider",
    "DatamuseProvider",
    "get_embedding_provider",
    "get_datamuse_provider",
]


def get_embedding_provider() -> EmbeddingProvider:
    """Get the configured embedding provider for contextual similarity.

    Environment variables:
        EMBEDDING_PROVIDER: "mock" (default) or "faiss"
        EMBEDDING_DATA_DIR: Path to FAISS data directory (required if provider=faiss)
    """
    import os

    provider_type = os.getenv("EMBEDDING_PROVIDER", "mock").lower()

    if provider_type == "mock":
        return MockEmbeddingProvider()

    elif provider_type == "faiss":
        from .faiss_provider import FAISSEmbeddingProvider

        data_dir = os.getenv("EMBEDDING_DATA_DIR")
        if not data_dir:
            raise ValueError(
                "EMBEDDING_DATA_DIR environment variable is required when using FAISS provider"
            )
        return FAISSEmbeddingProvider(data_dir)

    else:
        raise ValueError(f"Unknown embedding provider: {provider_type}")


def get_datamuse_provider() -> DatamuseProvider:
    """Get the Datamuse provider for semantic similarity."""
    return DatamuseProvider()
