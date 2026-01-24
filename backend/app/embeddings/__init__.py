"""Embedding service module."""

from .base import EmbeddingProvider
from .mock import MockEmbeddingProvider

__all__ = ["EmbeddingProvider", "MockEmbeddingProvider", "get_embedding_provider"]


def get_embedding_provider() -> EmbeddingProvider:
    """Get the configured embedding provider.

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
