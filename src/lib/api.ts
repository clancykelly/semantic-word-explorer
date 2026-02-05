import type { ExploreResponse, ErrorResponse, SearchMode, LayoutType } from "./types";

export type ApiResult =
  | { success: true; data: ExploreResponse }
  | { success: false; error: ErrorResponse };

export async function exploreWord(
  word: string,
  sense?: string,
  mode: SearchMode = "semantic",
  layout: LayoutType = "sectors",
  includeRare: boolean = false,
  relevance: number | null = null
): Promise<ApiResult> {
  try {
    const params = new URLSearchParams({ word, mode, layout });
    if (sense) {
      params.set("sense", sense);
    }
    if (includeRare) {
      params.set("includeRare", "true");
    }
    if (relevance !== null) {
      params.set("relevance", relevance.toString());
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

export async function expandCluster(
  word: string,
  anchorWords: string[],
  excludeWords: string[],
  limit: number = 50
): Promise<ApiResult> {
  try {
    const params = new URLSearchParams({
      word,
      anchor_words: anchorWords.join(","),
      limit: limit.toString(),
    });
    if (excludeWords.length > 0) {
      params.set("exclude", excludeWords.join(","));
    }

    const response = await fetch(`/api/expand-cluster?${params.toString()}`);
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
