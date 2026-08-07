"use client";

import { useMemo, useState } from "react";
import type { WordNeighbor, Cluster } from "@/lib/types";

const PILE_MARKER = "co-occurring";
const W = 1000;
const H = 640;
const MAX_LABELS = 44;

interface WordMapProps {
  neighbors: WordNeighbor[];
  clusters: Cluster[];
  queryWord: string;
  onExplore: (word: string) => void;
}

/**
 * A real semantic map: positions are a PCA projection of the words' actual
 * embedding vectors, so distance on screen means distance in meaning.
 */
export function WordMap({ neighbors, clusters, queryWord, onExplore }: WordMapProps) {
  const [hover, setHover] = useState<string | null>(null);

  const colorOf = useMemo(
    () => Object.fromEntries(clusters.map((c) => [c.id, c.color])),
    [clusters]
  );
  const pileIds = useMemo(
    () => new Set(clusters.filter((c) => c.label.includes(PILE_MARKER)).map((c) => c.id)),
    [clusters]
  );

  const q = queryWord.toLowerCase();
  const query = neighbors.find((n) => n.word.toLowerCase() === q);
  const words = useMemo(
    () => neighbors.filter((n) => n.word.toLowerCase() !== q),
    [neighbors, q]
  );
  // Greedy label placement: strongest words first, skip any label whose box
  // would collide with one already placed (dense zones stay legible; hidden
  // labels appear on hover).
  const labeled = useMemo(() => {
    type Box = { x1: number; y1: number; x2: number; y2: number };
    const placed: Box[] = [];
    if (query) {
      const qx = query.coordinates.x * W + 14;
      const qy = query.coordinates.y * H;
      placed.push({
        x1: qx - 18,
        y1: qy - 12,
        x2: qx + queryWord.length * 10.5,
        y2: qy + 12,
      });
    }
    const chosen = new Set<string>();
    const overlaps = (a: Box, b: Box) =>
      a.x1 < b.x2 && a.x2 > b.x1 && a.y1 < b.y2 && a.y2 > b.y1;
    for (const n of [...words].sort((a, b) => b.similarity - a.similarity)) {
      if (chosen.size >= MAX_LABELS) break;
      const fs = 9.5 + n.similarity * 5.5;
      const x = n.coordinates.x * W + 7;
      const y = n.coordinates.y * H;
      const box: Box = {
        x1: x - 8,
        y1: y - fs * 0.75,
        x2: x + n.word.length * fs * 0.58,
        y2: y + fs * 0.75,
      };
      if (placed.some((b) => overlaps(b, box))) continue;
      placed.push(box);
      chosen.add(n.word);
    }
    return chosen;
  }, [words, query, queryWord]);

  return (
    <div>
      <div className="border border-[var(--line)] rounded-[4px] bg-[var(--surface)] overflow-hidden">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block" role="img" aria-label={`Semantic map of words related to ${queryWord}`}>
          {words.map((n) => {
            const x = n.coordinates.x * W;
            const y = n.coordinates.y * H;
            const pile = pileIds.has(n.cluster);
            const active = hover === n.word;
            const show = labeled.has(n.word) || active;
            return (
              <g
                key={n.word}
                className="cursor-pointer"
                onClick={() => onExplore(n.word)}
                onMouseEnter={() => setHover(n.word)}
                onMouseLeave={() => setHover(null)}
              >
                <title>{`${n.word} · ${Math.round(n.similarity * 100)}%${pile ? " · co-occurring" : ""} · click to explore`}</title>
                {pile ? (
                  <circle
                    cx={x}
                    cy={y}
                    r={4}
                    fill="none"
                    stroke={colorOf[n.cluster]}
                    strokeWidth={1.4}
                    strokeDasharray="2.5 2"
                  />
                ) : (
                  <circle
                    cx={x}
                    cy={y}
                    r={2.6 + n.similarity * 3.4}
                    fill={colorOf[n.cluster]}
                    opacity={active ? 1 : 0.85}
                  />
                )}
                {show && (
                  <text
                    x={x + 7}
                    y={y + 3.5}
                    fontSize={9.5 + n.similarity * 5.5}
                    fontWeight={active ? 680 : 430 + n.similarity * 220}
                    fontStyle={(n.formality ?? 0.5) >= 0.6 ? "italic" : "normal"}
                    fill={pile ? "var(--muted)" : "var(--ink)"}
                  >
                    {n.word}
                  </text>
                )}
              </g>
            );
          })}
          {query && (
            <g>
              <circle
                cx={query.coordinates.x * W}
                cy={query.coordinates.y * H}
                r={11}
                fill="none"
                stroke="var(--ink)"
                strokeWidth={1}
                opacity={0.35}
              />
              <circle
                cx={query.coordinates.x * W}
                cy={query.coordinates.y * H}
                r={6.5}
                fill="var(--ink)"
              />
              <text
                x={query.coordinates.x * W + 14}
                y={query.coordinates.y * H + 4.5}
                fontSize={17}
                fontWeight={750}
                fill="var(--ink)"
              >
                {queryWord}
              </text>
            </g>
          )}
        </svg>
      </div>

      {/* Legend + hint */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-x-6 gap-y-2">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
          {clusters.map((c) => {
            const pile = c.label.includes(PILE_MARKER);
            return (
              <span key={c.id} className="inline-flex items-center gap-1.5 font-mono text-[10px] lowercase tracking-[0.06em] text-[var(--muted)]">
                <span
                  className="inline-block w-2 h-2 rounded-full"
                  style={
                    pile
                      ? { border: `1.4px dashed ${c.color}` }
                      : { backgroundColor: c.color }
                  }
                />
                {c.label}
              </span>
            );
          })}
        </div>
        <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-[var(--faint)]">
          distance = meaning · click a word to explore
        </p>
      </div>
    </div>
  );
}
