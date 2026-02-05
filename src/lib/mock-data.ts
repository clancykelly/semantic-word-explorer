// Mock data for development - simulates pre-computed embeddings
import type {
  WordNeighbor,
  Cluster,
  WordSense,
  ExploreResponse,
  FrequencyTier,
} from "./types";

// Cluster color palette
const CLUSTER_COLORS = [
  "#6366f1", // indigo
  "#ec4899", // pink
  "#14b8a6", // teal
  "#f59e0b", // amber
  "#8b5cf6", // violet
  "#06b6d4", // cyan
];

// Mock vocabulary with pre-computed relationships
// Each word has associated senses, neighbors by sense, and 2D coordinates

interface MockWordData {
  senses: WordSense[];
  neighborsBySense: Record<string, MockNeighborData[]>;
}

interface MockNeighborData {
  word: string;
  similarity: number;
  frequency: FrequencyTier;
  cluster: number;
  // Relative position from cluster centroid
  offsetX: number;
  offsetY: number;
}

// Cluster definitions for each sense
const CLUSTER_DEFINITIONS: Record<string, { label: string; centroid: { x: number; y: number } }[]> = {
  "happy|ADJ": [
    { label: "joy & delight", centroid: { x: 0.3, y: 0.7 } },
    { label: "satisfaction", centroid: { x: 0.7, y: 0.6 } },
    { label: "fortunate", centroid: { x: 0.5, y: 0.2 } },
  ],
  "bank|NOUN:financial": [
    { label: "institutions", centroid: { x: 0.2, y: 0.8 } },
    { label: "money & finance", centroid: { x: 0.6, y: 0.7 } },
    { label: "transactions", centroid: { x: 0.8, y: 0.3 } },
  ],
  "bank|NOUN:river": [
    { label: "water features", centroid: { x: 0.3, y: 0.6 } },
    { label: "landscape", centroid: { x: 0.7, y: 0.4 } },
  ],
  "run|VERB:move": [
    { label: "locomotion", centroid: { x: 0.2, y: 0.7 } },
    { label: "speed", centroid: { x: 0.6, y: 0.8 } },
    { label: "escape", centroid: { x: 0.8, y: 0.4 } },
  ],
  "run|VERB:operate": [
    { label: "manage", centroid: { x: 0.3, y: 0.6 } },
    { label: "function", centroid: { x: 0.7, y: 0.5 } },
  ],
  "light|NOUN": [
    { label: "illumination", centroid: { x: 0.3, y: 0.7 } },
    { label: "physics", centroid: { x: 0.7, y: 0.6 } },
    { label: "perception", centroid: { x: 0.5, y: 0.3 } },
  ],
  "light|ADJ": [
    { label: "weight", centroid: { x: 0.3, y: 0.7 } },
    { label: "brightness", centroid: { x: 0.7, y: 0.5 } },
    { label: "ease", centroid: { x: 0.5, y: 0.2 } },
  ],
  "ocean|NOUN": [
    { label: "water bodies", centroid: { x: 0.2, y: 0.7 } },
    { label: "marine life", centroid: { x: 0.5, y: 0.5 } },
    { label: "vastness", centroid: { x: 0.8, y: 0.6 } },
    { label: "journey", centroid: { x: 0.6, y: 0.2 } },
  ],
  "bright|ADJ": [
    { label: "luminous", centroid: { x: 0.3, y: 0.7 } },
    { label: "intelligent", centroid: { x: 0.7, y: 0.6 } },
    { label: "vivid", centroid: { x: 0.5, y: 0.3 } },
  ],
};

const MOCK_VOCABULARY: Record<string, MockWordData> = {
  happy: {
    senses: [
      { sense: "happy|ADJ", label: "happy (feeling joy)", frequency: 100 },
    ],
    neighborsBySense: {
      "happy|ADJ": [
        { word: "joyful", similarity: 0.92, frequency: "common", cluster: 0, offsetX: 0.05, offsetY: 0.03 },
        { word: "cheerful", similarity: 0.89, frequency: "common", cluster: 0, offsetX: -0.03, offsetY: 0.06 },
        { word: "delighted", similarity: 0.87, frequency: "common", cluster: 0, offsetX: 0.08, offsetY: -0.02 },
        { word: "elated", similarity: 0.84, frequency: "uncommon", cluster: 0, offsetX: -0.06, offsetY: 0.04 },
        { word: "jubilant", similarity: 0.81, frequency: "rare", cluster: 0, offsetX: 0.02, offsetY: 0.08 },
        { word: "blissful", similarity: 0.79, frequency: "uncommon", cluster: 0, offsetX: -0.04, offsetY: -0.05 },
        { word: "ecstatic", similarity: 0.77, frequency: "uncommon", cluster: 0, offsetX: 0.07, offsetY: 0.01 },
        { word: "content", similarity: 0.85, frequency: "common", cluster: 1, offsetX: 0.04, offsetY: -0.03 },
        { word: "satisfied", similarity: 0.82, frequency: "common", cluster: 1, offsetX: -0.02, offsetY: 0.05 },
        { word: "pleased", similarity: 0.80, frequency: "common", cluster: 1, offsetX: 0.06, offsetY: 0.02 },
        { word: "gratified", similarity: 0.75, frequency: "uncommon", cluster: 1, offsetX: -0.05, offsetY: -0.04 },
        { word: "fulfilled", similarity: 0.73, frequency: "uncommon", cluster: 1, offsetX: 0.03, offsetY: 0.06 },
        { word: "lucky", similarity: 0.71, frequency: "common", cluster: 2, offsetX: -0.02, offsetY: 0.04 },
        { word: "fortunate", similarity: 0.69, frequency: "common", cluster: 2, offsetX: 0.05, offsetY: -0.03 },
        { word: "blessed", similarity: 0.67, frequency: "common", cluster: 2, offsetX: -0.04, offsetY: 0.02 },
        { word: "serendipitous", similarity: 0.58, frequency: "rare", cluster: 2, offsetX: 0.07, offsetY: 0.05 },
      ],
    },
  },
  bank: {
    senses: [
      { sense: "bank|NOUN:financial", label: "bank (financial institution)", frequency: 80 },
      { sense: "bank|NOUN:river", label: "bank (edge of river)", frequency: 20 },
    ],
    neighborsBySense: {
      "bank|NOUN:financial": [
        { word: "institution", similarity: 0.88, frequency: "common", cluster: 0, offsetX: 0.04, offsetY: 0.02 },
        { word: "lender", similarity: 0.85, frequency: "common", cluster: 0, offsetX: -0.03, offsetY: 0.05 },
        { word: "treasury", similarity: 0.82, frequency: "uncommon", cluster: 0, offsetX: 0.06, offsetY: -0.02 },
        { word: "vault", similarity: 0.78, frequency: "common", cluster: 0, offsetX: -0.05, offsetY: 0.03 },
        { word: "money", similarity: 0.91, frequency: "common", cluster: 1, offsetX: 0.02, offsetY: 0.04 },
        { word: "savings", similarity: 0.87, frequency: "common", cluster: 1, offsetX: -0.04, offsetY: 0.01 },
        { word: "loan", similarity: 0.84, frequency: "common", cluster: 1, offsetX: 0.05, offsetY: -0.03 },
        { word: "mortgage", similarity: 0.80, frequency: "common", cluster: 1, offsetX: -0.02, offsetY: 0.06 },
        { word: "credit", similarity: 0.79, frequency: "common", cluster: 1, offsetX: 0.03, offsetY: 0.02 },
        { word: "deposit", similarity: 0.86, frequency: "common", cluster: 2, offsetX: -0.03, offsetY: 0.04 },
        { word: "withdrawal", similarity: 0.83, frequency: "common", cluster: 2, offsetX: 0.04, offsetY: -0.02 },
        { word: "transfer", similarity: 0.81, frequency: "common", cluster: 2, offsetX: -0.05, offsetY: 0.01 },
        { word: "transaction", similarity: 0.78, frequency: "common", cluster: 2, offsetX: 0.02, offsetY: 0.05 },
      ],
      "bank|NOUN:river": [
        { word: "shore", similarity: 0.93, frequency: "common", cluster: 0, offsetX: 0.03, offsetY: 0.04 },
        { word: "riverbank", similarity: 0.91, frequency: "common", cluster: 0, offsetX: -0.02, offsetY: 0.02 },
        { word: "waterside", similarity: 0.85, frequency: "uncommon", cluster: 0, offsetX: 0.05, offsetY: -0.03 },
        { word: "embankment", similarity: 0.82, frequency: "uncommon", cluster: 0, offsetX: -0.04, offsetY: 0.05 },
        { word: "levee", similarity: 0.78, frequency: "uncommon", cluster: 0, offsetX: 0.02, offsetY: 0.01 },
        { word: "slope", similarity: 0.75, frequency: "common", cluster: 1, offsetX: -0.03, offsetY: 0.03 },
        { word: "hillside", similarity: 0.72, frequency: "common", cluster: 1, offsetX: 0.04, offsetY: -0.02 },
        { word: "edge", similarity: 0.70, frequency: "common", cluster: 1, offsetX: -0.05, offsetY: 0.04 },
        { word: "verge", similarity: 0.68, frequency: "uncommon", cluster: 1, offsetX: 0.01, offsetY: 0.06 },
      ],
    },
  },
  run: {
    senses: [
      { sense: "run|VERB:move", label: "run (move quickly)", frequency: 70 },
      { sense: "run|VERB:operate", label: "run (operate/manage)", frequency: 30 },
    ],
    neighborsBySense: {
      "run|VERB:move": [
        { word: "sprint", similarity: 0.91, frequency: "common", cluster: 0, offsetX: 0.04, offsetY: 0.02 },
        { word: "jog", similarity: 0.88, frequency: "common", cluster: 0, offsetX: -0.02, offsetY: 0.05 },
        { word: "dash", similarity: 0.86, frequency: "common", cluster: 0, offsetX: 0.05, offsetY: -0.03 },
        { word: "race", similarity: 0.84, frequency: "common", cluster: 0, offsetX: -0.04, offsetY: 0.01 },
        { word: "gallop", similarity: 0.79, frequency: "uncommon", cluster: 0, offsetX: 0.02, offsetY: 0.06 },
        { word: "fast", similarity: 0.85, frequency: "common", cluster: 1, offsetX: -0.03, offsetY: 0.03 },
        { word: "swift", similarity: 0.82, frequency: "common", cluster: 1, offsetX: 0.04, offsetY: -0.02 },
        { word: "quick", similarity: 0.80, frequency: "common", cluster: 1, offsetX: -0.05, offsetY: 0.04 },
        { word: "hasty", similarity: 0.75, frequency: "uncommon", cluster: 1, offsetX: 0.01, offsetY: 0.05 },
        { word: "flee", similarity: 0.78, frequency: "common", cluster: 2, offsetX: -0.02, offsetY: 0.02 },
        { word: "escape", similarity: 0.76, frequency: "common", cluster: 2, offsetX: 0.05, offsetY: -0.04 },
        { word: "bolt", similarity: 0.73, frequency: "common", cluster: 2, offsetX: -0.04, offsetY: 0.03 },
        { word: "abscond", similarity: 0.65, frequency: "rare", cluster: 2, offsetX: 0.03, offsetY: 0.06 },
      ],
      "run|VERB:operate": [
        { word: "manage", similarity: 0.89, frequency: "common", cluster: 0, offsetX: 0.03, offsetY: 0.04 },
        { word: "operate", similarity: 0.87, frequency: "common", cluster: 0, offsetX: -0.02, offsetY: 0.02 },
        { word: "direct", similarity: 0.84, frequency: "common", cluster: 0, offsetX: 0.05, offsetY: -0.03 },
        { word: "oversee", similarity: 0.81, frequency: "uncommon", cluster: 0, offsetX: -0.04, offsetY: 0.05 },
        { word: "administer", similarity: 0.78, frequency: "uncommon", cluster: 0, offsetX: 0.02, offsetY: 0.01 },
        { word: "function", similarity: 0.85, frequency: "common", cluster: 1, offsetX: -0.03, offsetY: 0.03 },
        { word: "work", similarity: 0.83, frequency: "common", cluster: 1, offsetX: 0.04, offsetY: -0.02 },
        { word: "execute", similarity: 0.80, frequency: "common", cluster: 1, offsetX: -0.05, offsetY: 0.04 },
        { word: "perform", similarity: 0.77, frequency: "common", cluster: 1, offsetX: 0.01, offsetY: 0.06 },
      ],
    },
  },
  ocean: {
    senses: [
      { sense: "ocean|NOUN", label: "ocean (large body of water)", frequency: 100 },
    ],
    neighborsBySense: {
      "ocean|NOUN": [
        { word: "sea", similarity: 0.95, frequency: "common", cluster: 0, offsetX: 0.03, offsetY: 0.02 },
        { word: "atlantic", similarity: 0.88, frequency: "common", cluster: 0, offsetX: -0.02, offsetY: 0.05 },
        { word: "pacific", similarity: 0.87, frequency: "common", cluster: 0, offsetX: 0.05, offsetY: -0.01 },
        { word: "waters", similarity: 0.84, frequency: "common", cluster: 0, offsetX: -0.04, offsetY: 0.03 },
        { word: "deep", similarity: 0.79, frequency: "common", cluster: 0, offsetX: 0.02, offsetY: 0.06 },
        { word: "whale", similarity: 0.75, frequency: "common", cluster: 1, offsetX: -0.03, offsetY: 0.02 },
        { word: "dolphin", similarity: 0.72, frequency: "common", cluster: 1, offsetX: 0.04, offsetY: -0.03 },
        { word: "coral", similarity: 0.69, frequency: "common", cluster: 1, offsetX: -0.05, offsetY: 0.04 },
        { word: "fish", similarity: 0.68, frequency: "common", cluster: 1, offsetX: 0.02, offsetY: 0.01 },
        { word: "marine", similarity: 0.76, frequency: "common", cluster: 1, offsetX: -0.01, offsetY: 0.05 },
        { word: "vast", similarity: 0.71, frequency: "common", cluster: 2, offsetX: 0.03, offsetY: -0.02 },
        { word: "endless", similarity: 0.67, frequency: "common", cluster: 2, offsetX: -0.04, offsetY: 0.03 },
        { word: "immense", similarity: 0.65, frequency: "uncommon", cluster: 2, offsetX: 0.05, offsetY: 0.04 },
        { word: "fathomless", similarity: 0.58, frequency: "rare", cluster: 2, offsetX: -0.02, offsetY: 0.06 },
        { word: "voyage", similarity: 0.62, frequency: "uncommon", cluster: 3, offsetX: 0.04, offsetY: -0.01 },
        { word: "sail", similarity: 0.60, frequency: "common", cluster: 3, offsetX: -0.03, offsetY: 0.02 },
        { word: "horizon", similarity: 0.55, frequency: "common", cluster: 3, offsetX: 0.02, offsetY: 0.05 },
        { word: "tide", similarity: 0.73, frequency: "common", cluster: 0, offsetX: -0.05, offsetY: -0.02 },
        { word: "wave", similarity: 0.78, frequency: "common", cluster: 0, offsetX: 0.01, offsetY: 0.04 },
        { word: "salt", similarity: 0.52, frequency: "common", cluster: 1, offsetX: 0.06, offsetY: -0.04 },
      ],
    },
  },
  bright: {
    senses: [
      { sense: "bright|ADJ", label: "bright (emitting light / intelligent)", frequency: 100 },
    ],
    neighborsBySense: {
      "bright|ADJ": [
        { word: "luminous", similarity: 0.90, frequency: "uncommon", cluster: 0, offsetX: 0.03, offsetY: 0.04 },
        { word: "radiant", similarity: 0.88, frequency: "uncommon", cluster: 0, offsetX: -0.02, offsetY: 0.02 },
        { word: "brilliant", similarity: 0.86, frequency: "common", cluster: 0, offsetX: 0.05, offsetY: -0.03 },
        { word: "shining", similarity: 0.84, frequency: "common", cluster: 0, offsetX: -0.04, offsetY: 0.05 },
        { word: "gleaming", similarity: 0.81, frequency: "uncommon", cluster: 0, offsetX: 0.02, offsetY: 0.01 },
        { word: "incandescent", similarity: 0.75, frequency: "rare", cluster: 0, offsetX: -0.03, offsetY: 0.06 },
        { word: "clever", similarity: 0.82, frequency: "common", cluster: 1, offsetX: 0.04, offsetY: -0.02 },
        { word: "intelligent", similarity: 0.80, frequency: "common", cluster: 1, offsetX: -0.05, offsetY: 0.03 },
        { word: "smart", similarity: 0.79, frequency: "common", cluster: 1, offsetX: 0.01, offsetY: 0.05 },
        { word: "sharp", similarity: 0.76, frequency: "common", cluster: 1, offsetX: -0.02, offsetY: -0.04 },
        { word: "astute", similarity: 0.72, frequency: "uncommon", cluster: 1, offsetX: 0.05, offsetY: 0.02 },
        { word: "vivid", similarity: 0.85, frequency: "common", cluster: 2, offsetX: -0.03, offsetY: 0.04 },
        { word: "vibrant", similarity: 0.83, frequency: "common", cluster: 2, offsetX: 0.04, offsetY: -0.01 },
        { word: "colorful", similarity: 0.78, frequency: "common", cluster: 2, offsetX: -0.05, offsetY: 0.02 },
        { word: "striking", similarity: 0.74, frequency: "common", cluster: 2, offsetX: 0.02, offsetY: 0.06 },
      ],
    },
  },
  light: {
    senses: [
      { sense: "light|NOUN", label: "light (illumination)", frequency: 60 },
      { sense: "light|ADJ", label: "light (not heavy / bright)", frequency: 40 },
    ],
    neighborsBySense: {
      "light|NOUN": [
        { word: "illumination", similarity: 0.91, frequency: "uncommon", cluster: 0, offsetX: 0.03, offsetY: 0.02 },
        { word: "brightness", similarity: 0.89, frequency: "common", cluster: 0, offsetX: -0.02, offsetY: 0.05 },
        { word: "glow", similarity: 0.86, frequency: "common", cluster: 0, offsetX: 0.05, offsetY: -0.03 },
        { word: "radiance", similarity: 0.83, frequency: "uncommon", cluster: 0, offsetX: -0.04, offsetY: 0.01 },
        { word: "beam", similarity: 0.80, frequency: "common", cluster: 0, offsetX: 0.02, offsetY: 0.06 },
        { word: "photon", similarity: 0.75, frequency: "uncommon", cluster: 1, offsetX: -0.03, offsetY: 0.03 },
        { word: "wavelength", similarity: 0.72, frequency: "uncommon", cluster: 1, offsetX: 0.04, offsetY: -0.02 },
        { word: "spectrum", similarity: 0.70, frequency: "uncommon", cluster: 1, offsetX: -0.05, offsetY: 0.04 },
        { word: "ray", similarity: 0.78, frequency: "common", cluster: 1, offsetX: 0.01, offsetY: 0.05 },
        { word: "vision", similarity: 0.68, frequency: "common", cluster: 2, offsetX: -0.02, offsetY: 0.02 },
        { word: "sight", similarity: 0.66, frequency: "common", cluster: 2, offsetX: 0.05, offsetY: -0.04 },
        { word: "clarity", similarity: 0.64, frequency: "common", cluster: 2, offsetX: -0.04, offsetY: 0.03 },
      ],
      "light|ADJ": [
        { word: "lightweight", similarity: 0.92, frequency: "common", cluster: 0, offsetX: 0.03, offsetY: 0.04 },
        { word: "featherweight", similarity: 0.85, frequency: "uncommon", cluster: 0, offsetX: -0.02, offsetY: 0.02 },
        { word: "airy", similarity: 0.82, frequency: "common", cluster: 0, offsetX: 0.05, offsetY: -0.03 },
        { word: "weightless", similarity: 0.80, frequency: "uncommon", cluster: 0, offsetX: -0.04, offsetY: 0.05 },
        { word: "bright", similarity: 0.88, frequency: "common", cluster: 1, offsetX: 0.02, offsetY: 0.01 },
        { word: "pale", similarity: 0.79, frequency: "common", cluster: 1, offsetX: -0.03, offsetY: 0.03 },
        { word: "fair", similarity: 0.75, frequency: "common", cluster: 1, offsetX: 0.04, offsetY: -0.02 },
        { word: "easy", similarity: 0.73, frequency: "common", cluster: 2, offsetX: -0.05, offsetY: 0.04 },
        { word: "simple", similarity: 0.70, frequency: "common", cluster: 2, offsetX: 0.01, offsetY: 0.06 },
        { word: "gentle", similarity: 0.68, frequency: "common", cluster: 2, offsetX: -0.02, offsetY: -0.03 },
      ],
    },
  },
};

// Simple typo correction dictionary
const TYPO_CORRECTIONS: Record<string, string> = {
  happyness: "happiness",
  hapiness: "happiness",
  happines: "happiness",
  hapy: "happy",
  ocaen: "ocean",
  oceam: "ocean",
  bankk: "bank",
  runn: "run",
  runing: "running",
  brigth: "bright",
  brite: "bright",
  lite: "light",
  ligth: "light",
};

// Calculate Levenshtein distance for typo detection
function levenshteinDistance(a: string, b: string): number {
  const matrix: number[][] = [];

  for (let i = 0; i <= b.length; i++) {
    matrix[i] = [i];
  }
  for (let j = 0; j <= a.length; j++) {
    matrix[0][j] = j;
  }

  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      if (b.charAt(i - 1) === a.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j] + 1
        );
      }
    }
  }

  return matrix[b.length][a.length];
}

// Find closest word in vocabulary
function findClosestWord(input: string): string | null {
  // Check explicit typo corrections first
  if (TYPO_CORRECTIONS[input]) {
    return TYPO_CORRECTIONS[input];
  }

  // Check vocabulary with Levenshtein distance
  const words = Object.keys(MOCK_VOCABULARY);
  let closest: string | null = null;
  let minDistance = Infinity;

  for (const word of words) {
    const distance = levenshteinDistance(input, word);
    if (distance <= 2 && distance < minDistance) {
      minDistance = distance;
      closest = word;
    }
  }

  return closest;
}

// Generate mock response for a word and sense
export function getMockExploreResponse(
  word: string,
  requestedSense?: string
): ExploreResponse | { error: string; type: string; didYouMean?: string; suggestions?: string[] } {
  // Normalize input
  const normalizedWord = word.toLowerCase().trim().replace(/[^a-z]/g, "");

  if (!normalizedWord) {
    return {
      error: "Please enter a valid word",
      type: "invalid_input",
    };
  }

  // Check if word contains spaces (multi-word)
  if (word.trim().includes(" ")) {
    return {
      error: "Please enter a single word",
      type: "invalid_input",
    };
  }

  const wordData = MOCK_VOCABULARY[normalizedWord];

  if (!wordData) {
    // Check for typo
    const suggestion = findClosestWord(normalizedWord);
    if (suggestion) {
      return {
        error: `Word "${normalizedWord}" not found`,
        type: "not_found",
        didYouMean: suggestion,
      };
    }

    // Suggest similar words from vocabulary
    const suggestions = Object.keys(MOCK_VOCABULARY).slice(0, 3);
    return {
      error: `Word "${normalizedWord}" not found in vocabulary`,
      type: "not_found",
      suggestions,
    };
  }

  // Determine which sense to use
  const senses = wordData.senses;
  let selectedSense = requestedSense;

  if (!selectedSense || !wordData.neighborsBySense[selectedSense]) {
    // Default to most frequent sense
    selectedSense = senses.sort((a, b) => b.frequency - a.frequency)[0].sense;
  }

  const neighbors = wordData.neighborsBySense[selectedSense];
  const clusterDefs = CLUSTER_DEFINITIONS[selectedSense] || [
    { label: "related", centroid: { x: 0.5, y: 0.5 } },
  ];

  // Build clusters with colors
  const clusters: Cluster[] = clusterDefs.map((def, idx) => ({
    id: idx,
    label: def.label,
    color: CLUSTER_COLORS[idx % CLUSTER_COLORS.length],
    centroid: def.centroid,
  }));

  // Calculate actual coordinates for neighbors
  const wordNeighbors: WordNeighbor[] = neighbors.map((n) => {
    const clusterDef = clusterDefs[n.cluster] || clusterDefs[0];
    return {
      word: n.word,
      similarity: n.similarity,
      frequency: n.frequency,
      cluster: n.cluster,
      formality: 0.5, // Mock data default
      coordinates: {
        x: clusterDef.centroid.x + n.offsetX,
        y: clusterDef.centroid.y + n.offsetY,
      },
    };
  });

  // Add the query word itself at center
  const queryCentroid = clusterDefs[0].centroid;
  wordNeighbors.unshift({
    word: normalizedWord,
    similarity: 1.0,
    frequency: "common",
    cluster: 0,
    formality: 0.5,
    coordinates: {
      x: queryCentroid.x,
      y: queryCentroid.y,
    },
  });

  return {
    query: {
      word: word,
      normalizedWord,
      sense: selectedSense,
      availableSenses: senses,
    },
    neighbors: wordNeighbors,
    clusters,
    meta: {
      totalResults: wordNeighbors.length,
      queryTimeMs: Math.floor(Math.random() * 50) + 20,
    },
  };
}

// Get list of available words (for autocomplete/testing)
export function getAvailableWords(): string[] {
  return Object.keys(MOCK_VOCABULARY);
}
