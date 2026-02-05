// API Types for Semantic Word Explorer

export type FrequencyTier = "common" | "uncommon" | "rare";

export type SearchMode = "semantic" | "contextual";

export type LayoutType = "sectors" | "rings" | "force" | "grid";

export interface Coordinates {
  x: number;
  y: number;
}

export interface WordNeighbor {
  word: string;
  similarity: number;
  coordinates: Coordinates;
  frequency: FrequencyTier;
  cluster: number;
  formality: number; // 0.0 = casual, 1.0 = formal
}

export interface Cluster {
  id: number;
  label: string;
  color: string;
  centroid: Coordinates;
}

export interface WordSense {
  sense: string; // e.g., "bank|NOUN"
  label: string; // e.g., "bank (financial institution)"
  frequency: number; // relative frequency, higher = more common
}

export interface QueryInfo {
  word: string;
  normalizedWord: string;
  sense: string | null;
  availableSenses: WordSense[];
}

export interface ExploreResponse {
  query: QueryInfo;
  neighbors: WordNeighbor[];
  clusters: Cluster[];
  meta: {
    totalResults: number;
    queryTimeMs: number;
  };
}

export interface ErrorResponse {
  error: string;
  type: "not_found" | "invalid_input" | "server_error";
  suggestions?: string[]; // For typo corrections or similar words
  didYouMean?: string; // For typo correction
}

export interface SearchState {
  status: "idle" | "loading" | "success" | "error";
  data: ExploreResponse | null;
  error: ErrorResponse | null;
}
