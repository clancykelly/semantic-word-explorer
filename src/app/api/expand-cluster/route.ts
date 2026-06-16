import { NextRequest, NextResponse } from "next/server";
import type { ErrorResponse } from "@/lib/types";
import {
  BACKEND_URL,
  transformResponse,
  transformError,
  type PythonExploreResponse,
  type PythonErrorResponse,
} from "@/lib/backend";

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const word = searchParams.get("word");
  const anchorWords = searchParams.get("anchor_words");
  const exclude = searchParams.get("exclude");
  const limit = searchParams.get("limit") || "50";

  if (!word || !anchorWords) {
    const errorResponse: ErrorResponse = {
      error: "Missing required parameters: word and anchor_words",
      type: "invalid_input",
    };
    return NextResponse.json(errorResponse, { status: 400 });
  }

  // Build URL for Python backend
  const backendParams = new URLSearchParams();
  backendParams.set("word", word);
  backendParams.set("anchor_words", anchorWords);
  backendParams.set("limit", limit);
  if (exclude) {
    backendParams.set("exclude", exclude);
  }

  try {
    const response = await fetch(
      `${BACKEND_URL}/expand-cluster?${backendParams.toString()}`,
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
