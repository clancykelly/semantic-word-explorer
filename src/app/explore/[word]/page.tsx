"use client";

import { useEffect, useState, useCallback, useRef, use } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { SearchInput } from "@/components/SearchInput";
import { SensePicker } from "@/components/SensePicker";
import { WordMap } from "@/components/WordMap";
import { ClusterListView } from "@/components/ClusterListView";
import { exploreWord, expandCluster } from "@/lib/api";
import type { SearchState, WordNeighbor } from "@/lib/types";

type ViewMode = "list" | "map";

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
  // "" = all senses; otherwise a sense id from the inventory (e.g. "bank|1").
  const [selectedSense, setSelectedSense] = useState<string>("");
  const [includeRare, setIncludeRare] = useState<boolean>(false);
  const [viewMode, setViewMode] = useState<ViewMode>("list");
  const [expandedClusters, setExpandedClusters] = useState<Record<number, WordNeighbor[]>>({});
  const [expandingCluster, setExpandingCluster] = useState<number | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const toastTimer = useRef<number | undefined>(undefined);
  // Guards against a slow earlier response clobbering a newer one
  const requestIdRef = useRef(0);

  const fetchWord = useCallback(
    async (searchWord: string, sense: string | undefined, searchIncludeRare: boolean) => {
      const requestId = ++requestIdRef.current;
      setState((prev) => ({ ...prev, status: "loading", error: null }));

      const result = await exploreWord(searchWord, sense, searchIncludeRare);
      if (requestId !== requestIdRef.current) return; // stale response

      if (result.success) {
        setState({ status: "success", data: result.data, error: null });
      } else {
        setState({ status: "error", data: null, error: result.error });
      }
    },
    []
  );

  // Single fetch effect: word, sense, and rare toggle all funnel here.
  useEffect(() => {
    void fetchWord(decodedWord, selectedSense || undefined, includeRare);
  }, [decodedWord, selectedSense, includeRare, fetchWord]);

  // Reset transient view state when the word changes
  useEffect(() => {
    setExpandedClusters({});
    setSelectedSense("");
  }, [decodedWord]);

  const handleSenseChange = useCallback((sense: string) => {
    setSelectedSense(sense);
  }, []);

  const handleSearch = useCallback(
    (newWord: string) => {
      setSelectedSense("");
      router.push(`/explore/${encodeURIComponent(newWord)}`);
    },
    [router]
  );

  const handleExplore = useCallback(
    (clickedWord: string) => {
      setSelectedSense("");
      router.push(`/explore/${encodeURIComponent(clickedWord)}`);
    },
    [router]
  );

  const handleCopy = useCallback((copiedWord: string) => {
    const done = () => {
      setToast(copiedWord);
      window.clearTimeout(toastTimer.current);
      toastTimer.current = window.setTimeout(() => setToast(null), 1600);
    };
    const fallback = () => {
      const ta = document.createElement("textarea");
      ta.value = copiedWord;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      try {
        document.execCommand("copy");
      } finally {
        ta.remove();
      }
      done();
    };
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(copiedWord).then(done).catch(fallback);
    } else {
      fallback();
    }
  }, []);

  const handleRetry = useCallback(() => {
    fetchWord(decodedWord, selectedSense || undefined, includeRare);
  }, [decodedWord, selectedSense, fetchWord, includeRare]);

  const handleExpandCluster = useCallback(
    async (clusterId: number) => {
      if (!state.data || expandingCluster !== null) {
        return;
      }
      const clusterWords = state.data.neighbors
        .filter((n) => n.cluster === clusterId)
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, 5)
        .map((n) => n.word);
      if (clusterWords.length === 0) {
        return;
      }
      const excludeWords = state.data.neighbors.map((n) => n.word);

      setExpandingCluster(clusterId);
      const result = await expandCluster(decodedWord, clusterWords, excludeWords, 30);
      setExpandingCluster(null);

      if (result.success && result.data.neighbors.length > 0) {
        setExpandedClusters((prev) => ({ ...prev, [clusterId]: result.data.neighbors }));
      }
    },
    [state.data, decodedWord, expandingCluster]
  );

  return (
    <div className="min-h-screen">
      {/* Chrome */}
      <header className="sticky top-0 z-10 bg-[var(--paper)]/90 backdrop-blur-sm border-b border-[var(--line)]">
        <div className="max-w-5xl mx-auto px-5 py-3 flex flex-wrap items-center gap-x-6 gap-y-2">
          <Link
            href="/"
            className="font-mono text-[10px] uppercase tracking-[0.28em] text-[var(--muted)] hover:text-[var(--ink)] transition-colors whitespace-nowrap"
          >
            semantic explorer
          </Link>
          <div className="flex-1 min-w-[220px] max-w-xl">
            <SearchInput
              onSearch={handleSearch}
              isLoading={state.status === "loading"}
              initialValue={decodedWord}
            />
          </div>
          <label
            className="flex items-center gap-1.5 cursor-pointer font-mono text-[10px] uppercase tracking-[0.16em] text-[var(--muted)]"
            title="include rare and uncommon words"
          >
            <input
              type="checkbox"
              checked={includeRare}
              onChange={(e) => setIncludeRare(e.target.checked)}
              className="w-3 h-3 accent-[var(--accent)]"
            />
            rare
          </label>
          <div className="flex items-center border border-[var(--line)] rounded-[3px] overflow-hidden">
            {(["list", "map"] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setViewMode(mode)}
                aria-pressed={viewMode === mode}
                className={`font-mono text-[10px] uppercase tracking-[0.16em] px-3 py-1.5 transition-colors ${
                  viewMode === mode
                    ? "bg-[var(--ink)] text-[var(--paper)]"
                    : "text-[var(--muted)] hover:text-[var(--ink)]"
                }`}
              >
                {mode}
              </button>
            ))}
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-5 py-10">
        {/* Loading */}
        {state.status === "loading" && (
          <div className="py-28 text-center">
            <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-[var(--muted)] animate-pulse">
              searching “{decodedWord}”
            </p>
          </div>
        )}

        {/* Error */}
        {state.status === "error" && state.error && (
          <div className="py-24 max-w-md mx-auto text-center space-y-4">
            <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-[var(--accent)]">
              {state.error.type === "not_found" ? "not found" : "something broke"}
            </p>
            <p className="text-[15px] text-[var(--muted)]">{state.error.error}</p>
            {state.error.didYouMean && (
              <p className="text-[15px]">
                did you mean{" "}
                <button
                  onClick={() => handleSearch(state.error!.didYouMean!)}
                  className="text-[var(--accent)] underline underline-offset-4 decoration-1"
                >
                  {state.error.didYouMean}
                </button>
                ?
              </p>
            )}
            {state.error.suggestions && state.error.suggestions.length > 0 && (
              <div className="flex justify-center gap-4">
                {state.error.suggestions.map((s) => (
                  <button
                    key={s}
                    onClick={() => handleSearch(s)}
                    className="text-[15px] text-[var(--muted)] hover:text-[var(--accent)] transition-colors"
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}
            {state.error.type === "server_error" && (
              <button
                onClick={handleRetry}
                className="font-mono text-[11px] uppercase tracking-[0.16em] border border-[var(--line)] rounded-[3px] px-4 py-2 hover:border-[var(--accent)] hover:text-[var(--accent)] transition-colors"
              >
                retry
              </button>
            )}
          </div>
        )}

        {/* Results */}
        {state.status === "success" && state.data && (
          <div className="space-y-8">
            <div className="space-y-4">
              <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
                <h1 className="text-[40px] md:text-[52px] leading-none font-bold tracking-[-0.03em]">
                  {state.data.query.normalizedWord}
                </h1>
                <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-[var(--faint)]">
                  {state.data.meta.totalResults} words · {state.data.meta.queryTimeMs} ms
                </p>
              </div>
              <SensePicker
                senses={state.data.query.availableSenses}
                selectedSense={selectedSense}
                onSelectSense={handleSenseChange}
              />
            </div>

            {viewMode === "map" ? (
              <WordMap
                neighbors={state.data.neighbors}
                clusters={state.data.clusters}
                queryWord={state.data.query.normalizedWord}
                onExplore={handleExplore}
              />
            ) : (
              <ClusterListView
                neighbors={state.data.neighbors}
                clusters={state.data.clusters}
                queryWord={state.data.query.normalizedWord}
                expandedClusters={expandedClusters}
                expandingCluster={expandingCluster}
                onExplore={handleExplore}
                onCopy={handleCopy}
                onExpandCluster={handleExpandCluster}
              />
            )}
          </div>
        )}
      </main>

      {/* Copy toast */}
      {toast && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-20">
          <p className="font-mono text-[11px] bg-[var(--ink)] text-[var(--paper)] rounded-[3px] px-3 py-1.5 shadow-lg">
            “{toast}” copied
          </p>
        </div>
      )}
    </div>
  );
}
