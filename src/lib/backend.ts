import type { ExploreResponse, ErrorResponse } from "@/lib/types";

// Python backend URL - configurable via environment variable
export const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";

// snake_case response shape from the Python backend
export interface PythonExploreResponse {
  query: {
    word: string;
    normalized_word: string;
    sense: string | null;
    available_senses: Array<{
      sense: string;
      label: string;
      frequency: number;
    }>;
  };
  neighbors: Array<{
    word: string;
    similarity: number;
    coordinates: { x: number; y: number };
    frequency: string;
    cluster: number;
    formality: number;
  }>;
  clusters: Array<{
    id: number;
    label: string;
    color: string;
    centroid: { x: number; y: number };
  }>;
  meta: {
    total_results: number;
    query_time_ms: number;
  };
}

export interface PythonErrorResponse {
  error: string;
  type: string;
  did_you_mean?: string;
  suggestions?: string[];
}

export function transformResponse(python: PythonExploreResponse): ExploreResponse {
  return {
    query: {
      word: python.query.word,
      normalizedWord: python.query.normalized_word,
      sense: python.query.sense,
      availableSenses: python.query.available_senses,
    },
    neighbors: python.neighbors.map((n) => ({
      ...n,
      frequency: n.frequency as "common" | "uncommon" | "rare",
    })),
    clusters: python.clusters,
    meta: {
      totalResults: python.meta.total_results,
      queryTimeMs: python.meta.query_time_ms,
    },
  };
}

export function transformError(python: PythonErrorResponse): ErrorResponse {
  return {
    error: python.error,
    type: python.type as "not_found" | "invalid_input" | "server_error",
    ...(python.did_you_mean && { didYouMean: python.did_you_mean }),
    ...(python.suggestions && { suggestions: python.suggestions }),
  };
}
