import { NextRequest, NextResponse } from "next/server";
import { getMockExploreResponse } from "@/lib/mock-data";
import type { ExploreResponse, ErrorResponse } from "@/lib/types";

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const word = searchParams.get("word");
  const sense = searchParams.get("sense") || undefined;

  // Validate input
  if (!word) {
    const error: ErrorResponse = {
      error: "Missing required parameter: word",
      type: "invalid_input",
    };
    return NextResponse.json(error, { status: 400 });
  }

  // Check for multi-word input
  if (word.trim().includes(" ")) {
    const error: ErrorResponse = {
      error: "Please enter a single word",
      type: "invalid_input",
    };
    return NextResponse.json(error, { status: 400 });
  }

  // Simulate network latency for realistic feel
  await new Promise((resolve) => setTimeout(resolve, 100 + Math.random() * 200));

  // Get mock response
  const result = getMockExploreResponse(word, sense);

  // Check if it's an error response
  if ("error" in result) {
    const error: ErrorResponse = {
      error: result.error,
      type: result.type as "not_found" | "invalid_input" | "server_error",
      ...(result.didYouMean && { didYouMean: result.didYouMean }),
      ...(result.suggestions && { suggestions: result.suggestions }),
    };
    return NextResponse.json(error, { status: result.type === "not_found" ? 404 : 400 });
  }

  return NextResponse.json(result as ExploreResponse);
}
