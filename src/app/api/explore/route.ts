import { NextRequest, NextResponse } from "next/server";
import type { ExploreResponse, ErrorResponse } from "@/lib/types";

// Python backend URL - configurable via environment variable
const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";

// Transform snake_case response from Python to camelCase for frontend
interface PythonExploreResponse {
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

interface PythonErrorResponse {
  error: string;
  type: string;
  did_you_mean?: string;
  suggestions?: string[];
}

function transformResponse(python: PythonExploreResponse): ExploreResponse {
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

function transformError(python: PythonErrorResponse): ErrorResponse {
  return {
    error: python.error,
    type: python.type as "not_found" | "invalid_input" | "server_error",
    ...(python.did_you_mean && { didYouMean: python.did_you_mean }),
    ...(python.suggestions && { suggestions: python.suggestions }),
  };
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const word = searchParams.get("word");
  const sense = searchParams.get("sense");
  const mode = searchParams.get("mode") || "semantic";
  const layout = searchParams.get("layout") || "sectors";
  const includeRare = searchParams.get("includeRare") === "true";
  const relevance = searchParams.get("relevance");

  // Build URL for Python backend
  const backendParams = new URLSearchParams();
  if (word) backendParams.set("word", word);
  if (sense) backendParams.set("sense", sense);
  backendParams.set("mode", mode);
  backendParams.set("layout", layout);
  backendParams.set("include_rare", includeRare ? "true" : "false");
  if (relevance) backendParams.set("relevance", relevance);

  try {
    const response = await fetch(
      `${BACKEND_URL}/explore?${backendParams.toString()}`,
      {
        headers: {
          Accept: "application/json",
        },
      }
    );

    const data = await response.json();

    if (!response.ok) {
      // Transform and return error response
      const error = transformError(data as PythonErrorResponse);
      return NextResponse.json(error, { status: response.status });
    }

    // Transform and return success response
    const result = transformResponse(data as PythonExploreResponse);
    return NextResponse.json(result);
  } catch {
    // Backend unavailable - return error
    const errorResponse: ErrorResponse = {
      error: "Backend service unavailable. Please try again later.",
      type: "server_error",
    };
    return NextResponse.json(errorResponse, { status: 503 });
  }
}
