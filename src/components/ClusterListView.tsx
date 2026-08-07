"use client";

import { useMemo, useState } from "react";
import type { WordNeighbor, Cluster } from "@/lib/types";

type SortMode = "similarity" | "formality" | "alphabetical";

const PILE_MARKER = "co-occurring";

interface ClusterListViewProps {
  neighbors: WordNeighbor[];
  clusters: Cluster[];
  queryWord: string;
  expandedClusters: Record<number, WordNeighbor[]>;
  expandingCluster: number | null;
  onExplore: (word: string) => void;
  onCopy: (word: string) => void;
  onExpandCluster: (clusterId: number) => void;
}

export function ClusterListView({
  neighbors,
  clusters,
  queryWord,
  expandedClusters,
  expandingCluster,
  onExplore,
  onCopy,
  onExpandCluster,
}: ClusterListViewProps) {
  const [sortMode, setSortMode] = useState<SortMode>("similarity");

  const sections = useMemo(() => {
    const byCluster: Record<number, WordNeighbor[]> = {};
    clusters.forEach((c) => {
      byCluster[c.id] = [];
    });
    neighbors.forEach((n) => {
      if (n.word.toLowerCase() === queryWord.toLowerCase()) return;
      byCluster[n.cluster]?.push(n);
    });
    // Fold expanded words into their sections (deduped)
    Object.entries(expandedClusters).forEach(([cid, extra]) => {
      const id = Number(cid);
      if (!byCluster[id]) return;
      const seen = new Set(byCluster[id].map((n) => n.word.toLowerCase()));
      extra.forEach((n) => {
        if (!seen.has(n.word.toLowerCase())) {
          seen.add(n.word.toLowerCase());
          byCluster[id].push({ ...n, cluster: id });
        }
      });
    });

    const sorter = (a: WordNeighbor, b: WordNeighbor) => {
      switch (sortMode) {
        case "formality":
          return (b.formality ?? 0.5) - (a.formality ?? 0.5);
        case "alphabetical":
          return a.word.localeCompare(b.word);
        default:
          return b.similarity - a.similarity;
      }
    };
    Object.values(byCluster).forEach((group) => group.sort(sorter));

    return clusters
      .map((c) => ({
        cluster: c,
        isPile: c.label.includes(PILE_MARKER),
        words: byCluster[c.id] ?? [],
      }))
      .filter((s) => s.words.length > 0)
      .sort((a, b) => {
        if (a.isPile !== b.isPile) return a.isPile ? 1 : -1;
        return (b.words[0]?.similarity ?? 0) - (a.words[0]?.similarity ?? 0);
      });
  }, [neighbors, clusters, queryWord, expandedClusters, sortMode]);

  return (
    <div className="space-y-8">
      {/* Controls */}
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2">
        <div className="flex items-baseline gap-3 font-mono text-[10px] uppercase tracking-[0.14em]">
          <span className="text-[var(--faint)]">sort</span>
          {(["similarity", "formality", "alphabetical"] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => setSortMode(mode)}
              className={
                sortMode === mode
                  ? "text-[var(--ink)] border-b border-[var(--accent)]"
                  : "text-[var(--muted)] hover:text-[var(--ink)]"
              }
            >
              {mode === "similarity" ? "closeness" : mode === "alphabetical" ? "a–z" : mode}
            </button>
          ))}
        </div>
        <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-[var(--faint)]">
          weight = closeness · <span className="italic normal-case">italic</span> = formal ·{" "}
          <span className="text-[var(--accent)]">✦</span> rare · click opens · hover to copy
        </p>
      </div>

      {/* Sections */}
      {sections.map(({ cluster, isPile, words }) => {
        const isExpanded = (expandedClusters[cluster.id]?.length ?? 0) > 0;
        const isExpanding = expandingCluster === cluster.id;
        return (
          <section
            key={cluster.id}
            className={`border-l-2 pl-5 ${isPile ? "border-dashed" : ""}`}
            style={{ borderColor: isPile ? "var(--line)" : cluster.color }}
          >
            <div className="flex items-baseline gap-3 mb-3">
              <h3 className="font-mono text-[11px] uppercase tracking-[0.16em] text-[var(--muted)]">
                {cluster.label}
              </h3>
              <span className="font-mono text-[10px] text-[var(--faint)]">{words.length}</span>
              {!isPile && (
                <button
                  onClick={() => onExpandCluster(cluster.id)}
                  disabled={isExpanding || isExpanded}
                  className="font-mono text-[10px] uppercase tracking-[0.12em] text-[var(--faint)] hover:text-[var(--accent)] transition-colors disabled:hover:text-[var(--faint)]"
                  title={isExpanded ? "expanded" : "pull more words from this neighborhood"}
                >
                  {isExpanding ? "…" : isExpanded ? "expanded" : "+ more"}
                </button>
              )}
            </div>
            <div className="flex flex-wrap items-baseline gap-x-5 gap-y-2.5">
              {words.map((n) => {
                const sim = Math.max(0, Math.min(1, n.similarity));
                const formal = (n.formality ?? 0.5) >= 0.6;
                return (
                  <span key={n.word} className="group inline-flex items-baseline gap-1">
                    <button
                      onClick={() => onExplore(n.word)}
                      title={`open “${n.word}” · ${Math.round(sim * 100)}% close${formal ? " · formal" : ""}${n.frequency === "rare" ? " · rare" : ""}`}
                      className={`leading-snug transition-colors hover:text-[var(--accent)] ${
                        isPile ? "text-[var(--muted)]" : "text-[var(--ink)]"
                      }`}
                      style={{
                        fontWeight: isPile ? 430 : 420 + sim * 260,
                        fontSize: isPile ? "0.95rem" : `${0.95 + sim * 0.3}rem`,
                        fontStyle: formal ? "italic" : "normal",
                      }}
                    >
                      {n.word}
                      {n.frequency === "rare" && (
                        <sup className="text-[var(--accent)] text-[9px] ml-0.5 not-italic">✦</sup>
                      )}
                    </button>
                    <button
                      onClick={() => onCopy(n.word)}
                      title={`copy “${n.word}” to clipboard`}
                      aria-label={`copy ${n.word}`}
                      className="font-mono text-[9px] uppercase tracking-[0.08em] text-[var(--faint)] opacity-0 group-hover:opacity-100 focus:opacity-100 hover:text-[var(--accent)] transition-opacity"
                    >
                      copy
                    </button>
                  </span>
                );
              })}
            </div>
          </section>
        );
      })}
    </div>
  );
}
