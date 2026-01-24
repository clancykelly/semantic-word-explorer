"""FastAPI application for Semantic Word Explorer."""

import re
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .embeddings import EmbeddingProvider, get_embedding_provider
from .models import (
    Cluster,
    Coordinates,
    ErrorResponse,
    ErrorType,
    ExploreResponse,
    FrequencyTier,
    MetaInfo,
    QueryInfo,
    WordNeighbor,
    WordSense,
)

# Global embedding provider instance
embedding_provider: EmbeddingProvider | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global embedding_provider

    # Initialize embedding provider based on configuration
    # Set EMBEDDING_PROVIDER=faiss and EMBEDDING_DATA_DIR=/path/to/data for production
    embedding_provider = get_embedding_provider()
    print(f"Initialized embedding provider with {len(embedding_provider.get_available_words())} words")

    yield

    # Cleanup
    embedding_provider = None


app = FastAPI(
    title="Semantic Word Explorer API",
    description="Find semantically related words using word embeddings",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def normalize_word(word: str) -> str:
    """Normalize input word: lowercase, strip, remove non-alpha."""
    return re.sub(r"[^a-z]", "", word.lower().strip())


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "semantic-word-explorer-api",
        "version": "0.1.0",
    }


@app.get("/words")
async def list_words():
    """List all available words in the vocabulary."""
    if not embedding_provider:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return {"words": embedding_provider.get_available_words()}


@app.get(
    "/explore",
    response_model=ExploreResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        404: {"model": ErrorResponse, "description": "Word not found"},
    },
)
async def explore_word(
    word: Annotated[str, Query(description="Word to explore")],
    sense: Annotated[str | None, Query(description="Specific sense to use")] = None,
    limit: Annotated[int, Query(ge=1, le=500, description="Max neighbors to return")] = 100,
):
    """Find semantically related words for the given input.

    Returns neighbors clustered by meaning with 2D coordinates for visualization.
    """
    if not embedding_provider:
        raise HTTPException(status_code=503, detail="Service not initialized")

    start_time = time.perf_counter()

    # Validate input
    if not word or not word.strip():
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="Missing required parameter: word",
                type=ErrorType.INVALID_INPUT,
            ).model_dump(),
        )

    # Check for multi-word input
    if " " in word.strip():
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="Please enter a single word",
                type=ErrorType.INVALID_INPUT,
            ).model_dump(),
        )

    normalized = normalize_word(word)

    if not normalized:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="Please enter a valid word",
                type=ErrorType.INVALID_INPUT,
            ).model_dump(),
        )

    # Search for the word
    result = embedding_provider.search(normalized, sense, limit)

    if not result:
        # Word not found - check for typo
        similar = embedding_provider.find_similar_word(normalized)

        if similar:
            return JSONResponse(
                status_code=404,
                content=ErrorResponse(
                    error=f'Word "{normalized}" not found',
                    type=ErrorType.NOT_FOUND,
                    did_you_mean=similar,
                ).model_dump(),
            )

        # No similar word found - provide suggestions
        suggestions = embedding_provider.get_suggestions(3)
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error=f'Word "{normalized}" not found in vocabulary',
                type=ErrorType.NOT_FOUND,
                suggestions=suggestions,
            ).model_dump(),
        )

    # Build response
    query_time_ms = int((time.perf_counter() - start_time) * 1000)

    neighbors = [
        WordNeighbor(
            word=n.word,
            similarity=n.similarity,
            coordinates=Coordinates(x=n.coordinates[0], y=n.coordinates[1]),
            frequency=FrequencyTier(n.frequency),
            cluster=n.cluster,
        )
        for n in result.neighbors
    ]

    clusters = [
        Cluster(
            id=c["id"],
            label=c["label"],
            color=c["color"],
            centroid=Coordinates(x=c["centroid"]["x"], y=c["centroid"]["y"]),
        )
        for c in result.clusters
    ]

    return ExploreResponse(
        query=QueryInfo(
            word=word,
            normalized_word=result.normalized_word,
            sense=result.sense,
            available_senses=[
                WordSense(sense=s.sense, label=s.label, frequency=s.frequency)
                for s in result.available_senses
            ],
        ),
        neighbors=neighbors,
        clusters=clusters,
        meta=MetaInfo(
            total_results=len(neighbors),
            query_time_ms=query_time_ms,
        ),
    )
