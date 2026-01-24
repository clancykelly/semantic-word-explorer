"use client";

import { useEffect, useState, useCallback, use } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { SearchInput } from "@/components/SearchInput";
import { SensePicker } from "@/components/SensePicker";
import { WordScatterPlot } from "@/components/WordScatterPlot";
import { exploreWord } from "@/lib/api";
import type { SearchState } from "@/lib/types";

interface ExplorePageProps {
  params: Promise<{ word: string }>;
}

export default function ExplorePage({ params }: ExplorePageProps) {
  const { word } = use(params);
  const router = useRouter();
  const decodedWord = decodeURIComponent(word);

  const [state, setState] = useState<SearchState>({
    status: "loading",
    data: null,
    error: null,
  });
  const [selectedSense, setSelectedSense] = useState<string | null>(null);

  // Fetch word data
  const fetchWord = useCallback(
    async (searchWord: string, sense?: string) => {
      setState((prev) => ({ ...prev, status: "loading", error: null }));

      const result = await exploreWord(searchWord, sense);

      if (result.success) {
        setState({
          status: "success",
          data: result.data,
          error: null,
        });
        // Set initial sense if not already set
        if (!sense && result.data.query.sense) {
          setSelectedSense(result.data.query.sense);
        }
      } else {
        setState({
          status: "error",
          data: null,
          error: result.error,
        });
      }
    },
    []
  );

  // Initial load
  useEffect(() => {
    void fetchWord(decodedWord);
  }, [decodedWord, fetchWord]);

  // Handle sense change
  const handleSenseChange = useCallback(
    (sense: string) => {
      setSelectedSense(sense);
      fetchWord(decodedWord, sense);
    },
    [decodedWord, fetchWord]
  );

  // Handle new search
  const handleSearch = useCallback(
    (newWord: string) => {
      setSelectedSense(null);
      router.push(`/explore/${encodeURIComponent(newWord)}`);
    },
    [router]
  );

  // Handle clicking a word in the visualization
  const handleWordClick = useCallback(
    (clickedWord: string) => {
      setSelectedSense(null);
      router.push(`/explore/${encodeURIComponent(clickedWord)}`);
    },
    [router]
  );

  // Handle retry
  const handleRetry = useCallback(() => {
    fetchWord(decodedWord, selectedSense || undefined);
  }, [decodedWord, selectedSense, fetchWord]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-zinc-50 to-white dark:from-zinc-950 dark:to-zinc-900">
      {/* Header */}
      <header className="sticky top-0 z-10 bg-white/80 dark:bg-zinc-900/80 backdrop-blur-sm border-b border-zinc-200 dark:border-zinc-800">
        <div className="max-w-6xl mx-auto px-4 py-4 flex flex-col md:flex-row items-center gap-4">
          <Link
            href="/"
            className="text-xl font-bold text-zinc-900 dark:text-zinc-100 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
          >
            Semantic Explorer
          </Link>
          <div className="flex-1 w-full md:w-auto">
            <SearchInput
              onSearch={handleSearch}
              isLoading={state.status === "loading"}
              initialValue={decodedWord}
              didYouMean={state.error?.didYouMean}
              onDidYouMeanClick={handleSearch}
            />
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-6xl mx-auto px-4 py-8">
        {/* Loading state */}
        {state.status === "loading" && (
          <div className="flex flex-col items-center justify-center py-24">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-indigo-200 border-t-indigo-600 mb-4" />
            <p className="text-zinc-600 dark:text-zinc-400">
              Exploring &ldquo;{decodedWord}&rdquo;...
            </p>
          </div>
        )}

        {/* Error state */}
        {state.status === "error" && state.error && (
          <div className="flex flex-col items-center justify-center py-24">
            <div className="w-16 h-16 mb-4 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
              <svg
                className="w-8 h-8 text-red-600 dark:text-red-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                />
              </svg>
            </div>
            <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100 mb-2">
              {state.error.type === "not_found"
                ? "Word Not Found"
                : "Something Went Wrong"}
            </h2>
            <p className="text-zinc-600 dark:text-zinc-400 mb-4 text-center max-w-md">
              {state.error.error}
            </p>

            {/* Did you mean suggestion */}
            {state.error.didYouMean && (
              <p className="mb-4">
                Did you mean{" "}
                <button
                  onClick={() => handleSearch(state.error!.didYouMean!)}
                  className="text-indigo-600 dark:text-indigo-400 hover:underline font-medium"
                >
                  {state.error.didYouMean}
                </button>
                ?
              </p>
            )}

            {/* Suggestions */}
            {state.error.suggestions && state.error.suggestions.length > 0 && (
              <div className="mb-4">
                <p className="text-sm text-zinc-500 mb-2">Try these words:</p>
                <div className="flex gap-2">
                  {state.error.suggestions.map((suggestion) => (
                    <button
                      key={suggestion}
                      onClick={() => handleSearch(suggestion)}
                      className="px-4 py-1.5 rounded-full text-sm font-medium
                        bg-zinc-100 dark:bg-zinc-800
                        text-zinc-700 dark:text-zinc-300
                        hover:bg-indigo-100 dark:hover:bg-indigo-900/30
                        hover:text-indigo-700 dark:hover:text-indigo-300
                        transition-colors"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Retry button for server errors */}
            {state.error.type === "server_error" && (
              <button
                onClick={handleRetry}
                className="px-6 py-2 rounded-full bg-indigo-600 text-white hover:bg-indigo-700 transition-colors"
              >
                Try Again
              </button>
            )}
          </div>
        )}

        {/* Success state */}
        {state.status === "success" && state.data && (
          <div className="space-y-6">
            {/* Word heading and sense picker */}
            <div className="text-center space-y-4">
              <h1 className="text-3xl md:text-4xl font-bold text-zinc-900 dark:text-zinc-100">
                {state.data.query.normalizedWord}
              </h1>

              <SensePicker
                senses={state.data.query.availableSenses}
                selectedSense={selectedSense || state.data.query.sense || ""}
                onSelectSense={handleSenseChange}
              />

              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                Found {state.data.meta.totalResults} related words in{" "}
                {state.data.meta.queryTimeMs}ms
              </p>
            </div>

            {/* Visualization */}
            <WordScatterPlot
              neighbors={state.data.neighbors}
              clusters={state.data.clusters}
              queryWord={state.data.query.normalizedWord}
              onWordClick={handleWordClick}
            />

            {/* Instructions */}
            <div className="text-center text-sm text-zinc-500 dark:text-zinc-400">
              <p>
                <strong>Click</strong> a word to explore it •{" "}
                <strong>Hover</strong> for details •{" "}
                <strong>Scroll</strong> to zoom • <strong>Drag</strong> to pan
              </p>
            </div>

            {/* Legend */}
            <div className="flex flex-wrap justify-center gap-4 text-sm">
              {state.data.clusters.map((cluster) => (
                <div key={cluster.id} className="flex items-center gap-2">
                  <span
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: cluster.color }}
                  />
                  <span className="text-zinc-600 dark:text-zinc-400">
                    {cluster.label}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
