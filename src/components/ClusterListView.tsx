"use client";

import { useMemo, useState } from "react";
import type { WordNeighbor, Cluster } from "@/lib/types";

type SortMode = "similarity" | "formality" | "alphabetical";

interface ClusterListViewProps {
  neighbors: WordNeighbor[];
  clusters: Cluster[];
  queryWord: string;
  onWordClick: (word: string) => void;
}

export function ClusterListView({
  neighbors,
  clusters,
  queryWord,
  onWordClick,
}: ClusterListViewProps) {
  const [sortMode, setSortMode] = useState<SortMode>("similarity");

  // Group neighbors by cluster and sort based on current mode
  const clusterGroups = useMemo(() => {
    const groups: Record<number, WordNeighbor[]> = {};

    // Initialize groups for all clusters
    clusters.forEach((cluster) => {
      groups[cluster.id] = [];
    });

    // Group neighbors by cluster
    neighbors.forEach((neighbor) => {
      if (groups[neighbor.cluster]) {
        groups[neighbor.cluster].push(neighbor);
      }
    });

    // Sort each group based on mode
    Object.values(groups).forEach((group) => {
      group.sort((a, b) => {
        switch (sortMode) {
          case "formality":
            return (b.formality ?? 0.5) - (a.formality ?? 0.5); // formal first
          case "alphabetical":
            return a.word.localeCompare(b.word);
          default:
            return b.similarity - a.similarity;
        }
      });
    });

    return groups;
  }, [neighbors, clusters, sortMode]);

  // Sort clusters by the highest similarity word in each cluster
  const sortedClusters = useMemo(() => {
    return [...clusters].sort((a, b) => {
      const aMaxSim = clusterGroups[a.id]?.[0]?.similarity ?? 0;
      const bMaxSim = clusterGroups[b.id]?.[0]?.similarity ?? 0;
      return bMaxSim - aMaxSim;
    });
  }, [clusters, clusterGroups]);

  // Convert hex color to rgba for background
  const hexToRgba = (hex: string, alpha: number): string => {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  };

  // Get formality label
  const getFormalityLabel = (f: number): string => {
    if (f >= 0.7) return "formal";
    if (f >= 0.5) return "neutral";
    return "casual";
  };

  // Map similarity to font weight (400-700)
  const getWeightFromSimilarity = (sim: number): number => {
    return Math.round(400 + sim * 300);
  };

  return (
    <div className="space-y-4">
      {/* Sort controls */}
      <div className="flex items-center justify-between px-1">
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-500 dark:text-zinc-400">Sort by:</span>
          <div className="flex gap-1">
            {(["similarity", "formality", "alphabetical"] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setSortMode(mode)}
                className={`px-2.5 py-1 text-xs rounded-md transition-colors ${
                  sortMode === mode
                    ? "bg-zinc-200 dark:bg-zinc-700 text-zinc-800 dark:text-zinc-100 font-medium"
                    : "text-zinc-500 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-zinc-800"
                }`}
              >
                {mode === "similarity" ? "strength" : mode}
              </button>
            ))}
          </div>
        </div>
        <div className="text-xs text-zinc-400 dark:text-zinc-500">
          <span className="font-semibold">bold</span> = stronger
          <span className="mx-2 text-zinc-300 dark:text-zinc-600">•</span>
          <span className="italic">italic</span> = formal
        </div>
      </div>

      {sortedClusters.map((cluster) => {
        const words = clusterGroups[cluster.id];
        if (!words || words.length === 0) return null;

        return (
          <div
            key={cluster.id}
            className="rounded-2xl overflow-hidden border transition-shadow hover:shadow-lg"
            style={{
              borderColor: hexToRgba(cluster.color, 0.3),
              backgroundColor: hexToRgba(cluster.color, 0.03),
            }}
          >
            {/* Cluster header */}
            <div
              className="px-5 py-3 flex items-center gap-3 border-b"
              style={{
                backgroundColor: hexToRgba(cluster.color, 0.08),
                borderColor: hexToRgba(cluster.color, 0.15),
              }}
            >
              <span
                className="w-3 h-3 rounded-full flex-shrink-0 ring-2 ring-white dark:ring-zinc-900"
                style={{ backgroundColor: cluster.color }}
              />
              <h3 className="font-semibold text-zinc-700 dark:text-zinc-200 tracking-tight">
                {cluster.label}
              </h3>
              <span className="text-sm text-zinc-400 dark:text-zinc-500 font-medium">
                {words.length}
              </span>
            </div>

            {/* Word list */}
            <div className="px-4 py-4">
              <div className="flex flex-wrap gap-2.5">
                {words.map((neighbor) => {
                  const isQueryWord = neighbor.word.toLowerCase() === queryWord.toLowerCase();
                  const formality = neighbor.formality ?? 0.5;
                  const isFormal = formality >= 0.6;

                  return (
                    <button
                      key={neighbor.word}
                      onClick={() => !isQueryWord && onWordClick(neighbor.word)}
                      disabled={isQueryWord}
                      className={`
                        px-3 py-1.5 rounded-full text-sm
                        transition-all duration-150
                        ${isFormal ? "italic" : ""}
                        ${
                          isQueryWord
                            ? "bg-zinc-200 dark:bg-zinc-700 text-zinc-500 dark:text-zinc-400 cursor-default border-2 border-zinc-300 dark:border-zinc-600"
                            : "hover:scale-105 hover:-translate-y-0.5 hover:shadow-md cursor-pointer border-2"
                        }
                      `}
                      style={
                        isQueryWord
                          ? { fontWeight: 500 }
                          : {
                              borderColor: cluster.color,
                              backgroundColor: hexToRgba(cluster.color, 0.08),
                              fontWeight: getWeightFromSimilarity(neighbor.similarity),
                            }
                      }
                      title={`Strength: ${(neighbor.similarity * 100).toFixed(0)}% • Tone: ${getFormalityLabel(formality)}${neighbor.frequency !== "common" ? ` • ${neighbor.frequency}` : ""}`}
                    >
                      <span className="text-zinc-700 dark:text-zinc-200">
                        {neighbor.word}
                      </span>
                      {neighbor.frequency === "rare" && (
                        <span className="ml-1 text-xs text-amber-600 dark:text-amber-400 not-italic">
                          ✦
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        );
      })}

      {/* Legend */}
      <div className="text-center text-xs text-zinc-400 dark:text-zinc-500 pt-2 pb-2">
        <p className="flex items-center justify-center gap-2 flex-wrap">
          <span>Click to explore</span>
          <span className="text-zinc-300 dark:text-zinc-600">•</span>
          <span className="text-amber-500 dark:text-amber-400">✦</span>
          <span>rare</span>
          <span className="text-zinc-300 dark:text-zinc-600">•</span>
          <span>Hover for details</span>
        </p>
      </div>
    </div>
  );
}
