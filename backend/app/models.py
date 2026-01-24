"""Pydantic models for API request/response schemas."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class FrequencyTier(str, Enum):
    """Word frequency classification."""

    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"


class Coordinates(BaseModel):
    """2D coordinates for visualization."""

    x: float = Field(..., ge=0, le=1)
    y: float = Field(..., ge=0, le=1)


class WordNeighbor(BaseModel):
    """A word with its relationship to the query word."""

    word: str
    similarity: float = Field(..., ge=0, le=1)
    coordinates: Coordinates
    frequency: FrequencyTier
    cluster: int = Field(..., ge=0)


class Cluster(BaseModel):
    """A semantic cluster grouping related words."""

    id: int
    label: str
    color: str
    centroid: Coordinates


class WordSense(BaseModel):
    """A specific meaning/sense of a word."""

    sense: str  # e.g., "bank|NOUN:financial"
    label: str  # e.g., "bank (financial institution)"
    frequency: int  # relative frequency, higher = more common


class QueryInfo(BaseModel):
    """Information about the processed query."""

    word: str
    normalized_word: str
    sense: str | None
    available_senses: list[WordSense]


class MetaInfo(BaseModel):
    """Metadata about the query response."""

    total_results: int
    query_time_ms: int


class ExploreResponse(BaseModel):
    """Successful response from the explore endpoint."""

    query: QueryInfo
    neighbors: list[WordNeighbor]
    clusters: list[Cluster]
    meta: MetaInfo


class ErrorType(str, Enum):
    """Types of errors that can occur."""

    NOT_FOUND = "not_found"
    INVALID_INPUT = "invalid_input"
    SERVER_ERROR = "server_error"


class ErrorResponse(BaseModel):
    """Error response from the API."""

    error: str
    type: ErrorType
    did_you_mean: str | None = None
    suggestions: list[str] | None = None
