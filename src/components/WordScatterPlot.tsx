"use client";

import { useMemo, useCallback } from "react";
import dynamic from "next/dynamic";
import type Plotly from "plotly.js";
import type { WordNeighbor, Cluster } from "@/lib/types";

// Dynamic import for Plotly to avoid SSR issues
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface WordScatterPlotProps {
  neighbors: WordNeighbor[];
  clusters: Cluster[];
  queryWord: string;
  onWordClick: (word: string) => void;
}

// Map frequency tiers to marker sizes
const FREQUENCY_SIZES: Record<string, number> = {
  common: 12,
  uncommon: 10,
  rare: 8,
};

export function WordScatterPlot({
  neighbors,
  clusters,
  queryWord,
  onWordClick,
}: WordScatterPlotProps) {
  // Group neighbors by cluster for coloring
  const traceData = useMemo(() => {
    // Create a trace for each cluster
    return clusters.map((cluster) => {
      const clusterNeighbors = neighbors.filter((n) => n.cluster === cluster.id);

      return {
        type: "scattergl",
        mode: "text+markers",
        name: cluster.label,
        x: clusterNeighbors.map((n) => n.coordinates.x),
        y: clusterNeighbors.map((n) => n.coordinates.y),
        text: clusterNeighbors.map((n) => n.word),
        customdata: clusterNeighbors.map((n) => ({
          word: n.word,
          similarity: n.similarity,
          frequency: n.frequency,
        })),
        textposition: "top center",
        textfont: {
          size: 11,
          color: "#374151",
        },
        marker: {
          size: clusterNeighbors.map((n) => {
            // Make query word larger
            if (n.word === queryWord) return 18;
            return FREQUENCY_SIZES[n.frequency] || 10;
          }),
          color: cluster.color,
          opacity: clusterNeighbors.map((n) => {
            // Query word is fully opaque, others based on similarity
            if (n.word === queryWord) return 1;
            return 0.5 + n.similarity * 0.5;
          }),
          line: {
            color: clusterNeighbors.map((n) =>
              n.word === queryWord ? "#1e1b4b" : "transparent"
            ),
            width: clusterNeighbors.map((n) => (n.word === queryWord ? 3 : 0)),
          },
        },
        hovertemplate:
          "<b>%{text}</b><br>" +
          "Similarity: %{customdata.similarity:.0%}<br>" +
          "Frequency: %{customdata.frequency}<br>" +
          "<extra></extra>",
      } as unknown as Plotly.Data;
    });
  }, [neighbors, clusters, queryWord]);

  const layout = useMemo(
    () => ({
      autosize: true,
      margin: { l: 40, r: 40, t: 40, b: 40 },
      showlegend: true,
      legend: {
        orientation: "h" as const,
        yanchor: "bottom" as const,
        y: 1.02,
        xanchor: "center" as const,
        x: 0.5,
        font: { size: 11 },
      },
      xaxis: {
        showgrid: false,
        zeroline: false,
        showticklabels: false,
        range: [-0.15, 1.15],
      },
      yaxis: {
        showgrid: false,
        zeroline: false,
        showticklabels: false,
        range: [-0.15, 1.15],
      },
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      hovermode: "closest" as const,
      dragmode: "pan" as const,
    }),
    []
  );

  const config: Partial<Plotly.Config> = useMemo(
    () => ({
      displayModeBar: true,
      modeBarButtonsToRemove: [
        "select2d",
        "lasso2d",
        "autoScale2d",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
        "toggleSpikelines",
      ],
      displaylogo: false,
      responsive: true,
      scrollZoom: true,
    }),
    []
  );

  const handleClick = useCallback(
    (event: Readonly<Plotly.PlotMouseEvent>) => {
      if (event.points && event.points.length > 0) {
        const point = event.points[0];
        const customdata = point.customdata as unknown as { word: string } | undefined;
        if (customdata?.word && customdata.word !== queryWord) {
          onWordClick(customdata.word);
        }
      }
    },
    [onWordClick, queryWord]
  );

  return (
    <div className="w-full h-[500px] md:h-[600px] bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-700 overflow-hidden">
      <Plot
        data={traceData}
        layout={layout}
        config={config}
        style={{ width: "100%", height: "100%" }}
        onClick={handleClick}
        useResizeHandler
      />
    </div>
  );
}
