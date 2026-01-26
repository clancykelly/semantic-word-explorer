"use client";

import { useEffect, useState, useCallback, use } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { SearchInput } from "@/components/SearchInput";
import { SensePicker } from "@/components/SensePicker";
import { WordScatterPlot } from "@/components/WordScatterPlot";
import { exploreWord, expandCluster } from "@/lib/api";
import type { SearchState, SearchMode, LayoutType, WordNeighbor } from "@/lib/types";

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
  const [mode, setMode] = useState<SearchMode>("semantic");
  const [layout, setLayout] = useState<LayoutType>("sectors");
  const [includeRare, setIncludeRare] = useState<boolean>(false);
  const [expandedClusters, setExpandedClusters] = useState<Record<number, WordNeighbor[]>>({});
  const [expandingCluster, setExpandingCluster] = useState<number | null>(null);

  // Fetch word data
  const fetchWord = useCallback(
    async (searchWord: string, sense: string | undefined, searchMode: SearchMode, searchLayout: LayoutType, searchIncludeRare: boolean) => {
      setState((prev) => ({ ...prev, status: "loading", error: null }));

      const result = await exploreWord(searchWord, sense, searchMode, searchLayout, searchIncludeRare);

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

  // Initial load and mode/layout/includeRare change
  useEffect(() => {
    void fetchWord(decodedWord, selectedSense || undefined, mode, layout, includeRare);
  }, [decodedWord, mode, layout, includeRare, fetchWord, selectedSense]);

  // Handle sense change
  const handleSenseChange = useCallback(
    (sense: string) => {
      setSelectedSense(sense);
      fetchWord(decodedWord, sense, mode, layout, includeRare);
    },
    [decodedWord, fetchWord, mode, layout, includeRare]
  );

  // Handle mode change
  const handleModeChange = useCallback(
    (newMode: SearchMode) => {
      setMode(newMode);
    },
    []
  );

  // Handle layout change
  const handleLayoutChange = useCallback(
    (newLayout: LayoutType) => {
      setLayout(newLayout);
    },
    []
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
    fetchWord(decodedWord, selectedSense || undefined, mode, layout, includeRare);
  }, [decodedWord, selectedSense, fetchWord, mode, layout, includeRare]);

  // Handle include rare toggle
  const handleIncludeRareChange = useCallback(
    (newIncludeRare: boolean) => {
      setIncludeRare(newIncludeRare);
    },
    []
  );

  // Handle cluster expansion
  const handleExpandCluster = useCallback(
    async (clusterId: number) => {
      if (!state.data || expandingCluster !== null) {
        return;
      }

      // Get anchor words from this cluster (top 5 most similar)
      const clusterWords = state.data.neighbors
        .filter((n) => n.cluster === clusterId)
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, 5)
        .map((n) => n.word);

      if (clusterWords.length === 0) {
        return;
      }

      // Get all currently shown words to exclude
      const excludeWords = state.data.neighbors.map((n) => n.word);

      setExpandingCluster(clusterId);

      const result = await expandCluster(
        decodedWord,
        clusterWords,
        excludeWords,
        30
      );

      setExpandingCluster(null);

      if (result.success && result.data.neighbors.length > 0) {
        setExpandedClusters((prev) => ({
          ...prev,
          [clusterId]: result.data.neighbors,
        }));
      }
    },
    [state.data, decodedWord, expandingCluster]
  );

  // Reset expanded clusters when word changes
  useEffect(() => {
    setExpandedClusters({});
  }, [decodedWord]);

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
          {/* Mode Toggle */}
          <div className="flex items-center gap-1 p-1 bg-zinc-100 dark:bg-zinc-800 rounded-full">
            <button
              onClick={() => handleModeChange("semantic")}
              className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all ${
                mode === "semantic"
                  ? "bg-white dark:bg-zinc-700 text-indigo-600 dark:text-indigo-400 shadow-sm"
                  : "text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200"
              }`}
              title="Find true synonyms and words with similar meanings"
            >
              Semantic
            </button>
            <button
              onClick={() => handleModeChange("contextual")}
              className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all ${
                mode === "contextual"
                  ? "bg-white dark:bg-zinc-700 text-indigo-600 dark:text-indigo-400 shadow-sm"
                  : "text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200"
              }`}
              title="Find words that appear in similar contexts"
            >
              Contextual
            </button>
          </div>

          {/* Layout Selector (semantic mode only) */}
          {mode === "semantic" && (
            <select
              value={layout}
              onChange={(e) => handleLayoutChange(e.target.value as LayoutType)}
              className="px-3 py-1.5 text-sm rounded-lg border border-zinc-200 dark:border-zinc-700
                bg-white dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300
                focus:outline-none focus:ring-2 focus:ring-indigo-500"
              title="Choose visualization layout"
            >
              <option value="sectors">Sectors</option>
              <option value="rings">Rings</option>
              <option value="force">Force</option>
              <option value="grid">Grid</option>
            </select>
          )}

          {/* Include Rare Words Toggle (semantic mode only) */}
          {mode === "semantic" && (
            <label
              className="flex items-center gap-2 cursor-pointer"
              title="Include rare and uncommon words in results"
            >
              <input
                type="checkbox"
                checked={includeRare}
                onChange={(e) => handleIncludeRareChange(e.target.checked)}
                className="w-4 h-4 rounded border-zinc-300 dark:border-zinc-600
                  text-indigo-600 focus:ring-indigo-500 focus:ring-offset-0"
              />
              <span className="text-sm text-zinc-600 dark:text-zinc-400">
                Rare words
              </span>
            </label>
          )}
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

              {/* Mode description */}
              <p className="text-xs text-zinc-400 dark:text-zinc-500 max-w-md mx-auto">
                {mode === "semantic" ? (
                  <>
                    <span className="font-medium text-indigo-600 dark:text-indigo-400">Semantic mode:</span>{" "}
                    True synonyms and words with similar meanings
                  </>
                ) : (
                  <>
                    <span className="font-medium text-indigo-600 dark:text-indigo-400">Contextual mode:</span>{" "}
                    Words that appear in similar contexts (co-occurrence)
                  </>
                )}
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

            {/* Legend with expand buttons */}
            <div className="flex flex-wrap justify-center gap-4 text-sm">
              {state.data.clusters.map((cluster) => {
                const isExpanded = expandedClusters[cluster.id]?.length > 0;
                const isExpanding = expandingCluster === cluster.id;

                return (
                  <div key={cluster.id} className="flex items-center gap-2">
                    <span
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: cluster.color }}
                    />
                    <span className="text-zinc-600 dark:text-zinc-400">
                      {cluster.label}
                    </span>
                    {mode === "semantic" && (
                      <button
                        onClick={() => handleExpandCluster(cluster.id)}
                        disabled={isExpanding || isExpanded}
                        className={`ml-1 px-2 py-0.5 text-xs rounded-full transition-colors ${
                          isExpanding
                            ? "bg-indigo-100 dark:bg-indigo-900/30 text-indigo-400 cursor-wait"
                            : isExpanded
                            ? "bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400"
                            : "bg-zinc-100 dark:bg-zinc-800 text-zinc-500 hover:bg-indigo-100 dark:hover:bg-indigo-900/30 hover:text-indigo-600 dark:hover:text-indigo-400"
                        }`}
                        title={isExpanded ? "Cluster expanded" : "Load more words from this cluster"}
                      >
                        {isExpanding ? "..." : isExpanded ? `+${expandedClusters[cluster.id].length}` : "+"}
                      </button>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Expanded cluster words */}
            {Object.keys(expandedClusters).length > 0 && (
              <div className="mt-6 p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded-xl border-2 border-indigo-200 dark:border-indigo-800">
                <h3 className="text-sm font-semibold text-indigo-700 dark:text-indigo-300 mb-3">
                  Expanded Results (click a word to explore)
                </h3>
                <div className="space-y-3">
                  {state.data.clusters.map((cluster) => {
                    const expanded = expandedClusters[cluster.id];
                    if (!expanded?.length) return null;

                    return (
                      <div key={cluster.id} className="flex flex-wrap items-center gap-2">
                        <span
                          className="w-2 h-2 rounded-full flex-shrink-0"
                          style={{ backgroundColor: cluster.color }}
                        />
                        <span className="text-xs text-zinc-500 dark:text-zinc-400 mr-1">
                          {cluster.label}:
                        </span>
                        {expanded.map((neighbor) => (
                          <button
                            key={neighbor.word}
                            onClick={() => handleWordClick(neighbor.word)}
                            className="px-2 py-0.5 text-xs rounded-full
                              bg-white dark:bg-zinc-700
                              text-zinc-700 dark:text-zinc-300
                              border border-zinc-200 dark:border-zinc-600
                              hover:border-indigo-300 dark:hover:border-indigo-500
                              hover:text-indigo-600 dark:hover:text-indigo-400
                              transition-colors"
                          >
                            {neighbor.word}
                          </button>
                        ))}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
