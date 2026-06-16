"""FastAPI application for Semantic Word Explorer."""

import os
import re
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .embeddings import DatamuseProvider, get_datamuse_provider, load_word_vectors
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

# Single semantic provider (Datamuse), optionally backed by static word vectors.
provider: DatamuseProvider | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global provider

    provider = get_datamuse_provider()
    print("Initialized semantic provider (Datamuse API)")

    # Load static word vectors (Model2Vec) for relevance filtering + clustering.
    # Optional: without them, clustering falls back to part-of-speech grouping.
    vectors = load_word_vectors()
    if vectors is not None:
        provider.set_vectors(vectors)
    else:
        print("Word vectors unavailable; clustering will fall back to part-of-speech grouping")

    yield

    provider = None


app = FastAPI(
    title="Semantic Word Explorer API",
    description="Find semantically related words using word embeddings",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS for frontend access (override origins via CORS_ORIGINS env var)
_default_origins = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001"
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", _default_origins).split(",") if o.strip()],
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


@app.get(
    "/explore",
    response_model=ExploreResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        404: {"model": ErrorResponse, "description": "Word not found"},
    },
)
def explore_word(
    word: Annotated[str, Query(description="Word to explore")],
    sense: Annotated[str | None, Query(description="Specific sense to use")] = None,
    limit: Annotated[int, Query(ge=1, le=500, description="Max neighbors to return")] = 150,
    layout: Annotated[str, Query(description="Visualization layout: 'sectors' or 'force'")] = "sectors",
    include_rare: Annotated[bool, Query(description="Include rare/uncommon words")] = False,
    relevance: Annotated[
        float | None,
        Query(ge=0.0, le=1.0, description="Relevance threshold (0.0-1.0). None=adaptive. Higher=stricter."),
    ] = None,
):
    """Find semantically related words for the given input.

    Returns neighbors clustered by meaning with 2D coordinates for visualization.

    Relevance:
    - None (default): adaptive threshold based on score distribution
    - 0.2: loose (more related words, some noise)
    - 0.3: moderate
    - 0.4+: strict (only close synonyms)
    """
    if not provider:
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
    result = provider.search(
        normalized, sense, limit,
        layout=layout, include_rare=include_rare, relevance=relevance,
    )

    if not result:
        # Word not found - check for typo
        similar = provider.find_similar_word(normalized)

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
        suggestions = provider.get_suggestions(3)
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
            formality=n.formality,
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


@app.get(
    "/expand-cluster",
    response_model=ExploreResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        404: {"model": ErrorResponse, "description": "No results found"},
    },
)
def expand_cluster(
    word: Annotated[str, Query(description="Original query word")],
    anchor_words: Annotated[str, Query(description="Comma-separated anchor words from cluster")],
    limit: Annotated[int, Query(ge=1, le=100, description="Max additional words")] = 50,
    exclude: Annotated[str | None, Query(description="Comma-separated words to exclude")] = None,
):
    """Expand a semantic cluster to fetch more related words.

    Given anchor words that represent a cluster (e.g., words from the "movement"
    sense of "run"), fetches additional words from that same semantic neighborhood.

    This enables drill-down exploration of polysemous words - first see the
    overview of all senses, then expand the sense you're interested in.
    """
    if not provider:
        raise HTTPException(status_code=503, detail="Service not initialized")

    start_time = time.perf_counter()

    # Parse and normalize anchor words
    anchors = [normalize_word(w) for w in anchor_words.split(",")]
    anchors = [w for w in anchors if w]  # Filter out empty after normalization
    if not anchors:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="At least one valid anchor word is required",
                type=ErrorType.INVALID_INPUT,
            ).model_dump(),
        )

    # Validate anchor word count
    if len(anchors) > 10:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="Maximum 10 anchor words allowed",
                type=ErrorType.INVALID_INPUT,
            ).model_dump(),
        )

    # Parse and normalize exclude words
    exclude_list = []
    if exclude:
        exclude_list = [normalize_word(w) for w in exclude.split(",")]
        exclude_list = [w for w in exclude_list if w]

    # Normalize the query word
    normalized_word = normalize_word(word)
    if not normalized_word:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="Please enter a valid word",
                type=ErrorType.INVALID_INPUT,
            ).model_dump(),
        )

    # Expand the cluster
    result = provider.expand_cluster(
        word=normalized_word,
        anchor_words=anchors,
        limit=limit,
        exclude_words=exclude_list,
    )

    if not result or not result.neighbors:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error="No additional words found for this cluster",
                type=ErrorType.NOT_FOUND,
            ).model_dump(),
        )

    query_time_ms = int((time.perf_counter() - start_time) * 1000)

    neighbors = [
        WordNeighbor(
            word=n.word,
            similarity=n.similarity,
            coordinates=Coordinates(x=n.coordinates[0], y=n.coordinates[1]),
            frequency=FrequencyTier(n.frequency),
            cluster=n.cluster,
            formality=n.formality,
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
            sense=None,
            available_senses=[],
        ),
        neighbors=neighbors,
        clusters=clusters,
        meta=MetaInfo(
            total_results=len(neighbors),
            query_time_ms=query_time_ms,
        ),
    )
