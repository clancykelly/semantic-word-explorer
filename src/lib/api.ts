import type { ExploreResponse, ErrorResponse } from "./types";

export type ApiResult =
  | { success: true; data: ExploreResponse }
  | { success: false; error: ErrorResponse };

export async function exploreWord(
  word: string,
  sense?: string
): Promise<ApiResult> {
  try {
    const params = new URLSearchParams({ word });
    if (sense) {
      params.set("sense", sense);
    }

    const response = await fetch(`/api/explore?${params.toString()}`);
    const data = await response.json();

    if (!response.ok) {
      return {
        success: false,
        error: data as ErrorResponse,
      };
    }

    return {
      success: true,
      data: data as ExploreResponse,
    };
  } catch {
    return {
      success: false,
      error: {
        error: "Network error. Please check your connection and try again.",
        type: "server_error",
      },
    };
  }
}
