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
  const sense = searchParams.get("sense");
  const includeRare = searchParams.get("includeRare") === "true";

  // Build URL for Python backend
  const backendParams = new URLSearchParams();
  if (word) backendParams.set("word", word);
  if (sense) backendParams.set("sense", sense);
  backendParams.set("include_rare", includeRare ? "true" : "false");

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
  } catch (err) {
    // Backend unavailable - return error
    console.error("explore route error:", err);
    const errorResponse: ErrorResponse = {
      error: "Backend service unavailable. Please try again later.",
      type: "server_error",
    };
    return NextResponse.json(errorResponse, { status: 503 });
  }
}
