"""FAISS-based embedding provider for production use.

This provider uses pre-computed GloVe embeddings with FAISS for fast
nearest neighbor search.

To use this provider:
1. Download GloVe embeddings (e.g., glove.6B.300d.txt)
2. Run the build_index.py script to create the FAISS index
3. Set EMBEDDING_PROVIDER=faiss in your environment
"""

import json
import os
from pathlib import Path

import numpy as np

from .base import EmbeddingProvider, SearchResult, SenseInfo, WordResult

# Only import FAISS if available (it's an optional dependency)
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

# Only import UMAP if available
try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    umap = None


class FAISSEmbeddingProvider(EmbeddingProvider):
    """Production embedding provider using FAISS for fast similarity search.

    Expected data directory structure:
        data/
        ├── embeddings.npy      # (N, D) float32 array of word embeddings
        ├── words.json          # List of words in same order as embeddings
        ├── word2idx.json       # Dict mapping word -> index
        ├── coordinates.npy     # (N, 2) float32 array of UMAP coordinates
        ├── clusters.json       # Cluster assignments and metadata
        └── faiss.index         # FAISS index file
    """

    def __init__(self, data_dir: str | Path):
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is not installed. Install with: pip install faiss-cpu"
            )

        self.data_dir = Path(data_dir)
        self._load_data()

    def _load_data(self):
        """Load embeddings, vocabulary, and FAISS index."""
        # Load word list
        with open(self.data_dir / "words.json") as f:
            self.words: list[str] = json.load(f)

        # Load word to index mapping
        with open(self.data_dir / "word2idx.json") as f:
            self.word2idx: dict[str, int] = json.load(f)

        # Load embeddings (memory-mapped for efficiency with large files)
        embeddings_path = self.data_dir / "embeddings.npy"
        self.embeddings = np.load(embeddings_path, mmap_mode="r")

        # Load UMAP coordinates
        coords_path = self.data_dir / "coordinates.npy"
        if coords_path.exists():
            self.coordinates = np.load(coords_path, mmap_mode="r")
        else:
            # Generate random coordinates if not available
            print("Warning: coordinates.npy not found, using random positions")
            self.coordinates = np.random.rand(len(self.words), 2).astype(np.float32)

        # Load cluster info
        clusters_path = self.data_dir / "clusters.json"
        if clusters_path.exists():
            with open(clusters_path) as f:
                self.cluster_data = json.load(f)
        else:
            self.cluster_data = {"assignments": [0] * len(self.words), "clusters": []}

        # Load or build FAISS index
        index_path = self.data_dir / "faiss.index"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        else:
            print("Building FAISS index (this may take a while)...")
            self.index = self._build_index()
            faiss.write_index(self.index, str(index_path))

        # Precompute word frequencies (simplified: based on index position)
        # In production, load actual frequency data
        self.frequencies = self._compute_frequencies()

        print(f"Loaded {len(self.words)} words with {self.embeddings.shape[1]}d embeddings")

    def _build_index(self) -> "faiss.Index":
        """Build FAISS index from embeddings."""
        d = self.embeddings.shape[1]
        n = self.embeddings.shape[0]

        # Normalize embeddings for cosine similarity
        embeddings_normalized = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )

        if n < 10000:
            # Small dataset: use flat index
            index = faiss.IndexFlatIP(d)
        else:
            # Large dataset: use IVF index for faster search
            nlist = min(int(np.sqrt(n)), 1000)
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings_normalized.astype(np.float32))

        index.add(embeddings_normalized.astype(np.float32))
        return index

    def _compute_frequencies(self) -> dict[str, str]:
        """Compute frequency tiers based on word index (proxy for frequency)."""
        frequencies = {}
        n = len(self.words)

        for i, word in enumerate(self.words):
            # Top 20% = common, next 30% = uncommon, rest = rare
            if i < n * 0.2:
                frequencies[word] = "common"
            elif i < n * 0.5:
                frequencies[word] = "uncommon"
            else:
                frequencies[word] = "rare"

        return frequencies

    def search(
        self,
        word: str,
        sense: str | None = None,
        limit: int = 100,
    ) -> SearchResult | None:
        normalized = word.lower().strip()

        if normalized not in self.word2idx:
            return None

        idx = self.word2idx[normalized]
        query_embedding = self.embeddings[idx : idx + 1].astype(np.float32)

        # Normalize for cosine similarity
        query_normalized = query_embedding / np.linalg.norm(query_embedding)

        # Search for nearest neighbors
        k = min(limit + 1, len(self.words))  # +1 to account for the word itself
        distances, indices = self.index.search(query_normalized, k)

        # Build neighbor list
        neighbors = []
        for dist, neighbor_idx in zip(distances[0], indices[0]):
            if neighbor_idx < 0:
                continue

            neighbor_word = self.words[neighbor_idx]
            coords = self.coordinates[neighbor_idx]
            cluster = self.cluster_data["assignments"][neighbor_idx]

            neighbors.append(
                WordResult(
                    word=neighbor_word,
                    similarity=float(dist),  # Inner product = cosine similarity
                    coordinates=(float(coords[0]), float(coords[1])),
                    frequency=self.frequencies.get(neighbor_word, "common"),
                    cluster=cluster,
                )
            )

        # Build cluster metadata
        clusters = []
        seen_clusters = set()
        for n in neighbors:
            if n.cluster not in seen_clusters:
                seen_clusters.add(n.cluster)
                cluster_info = next(
                    (c for c in self.cluster_data.get("clusters", []) if c["id"] == n.cluster),
                    {"id": n.cluster, "label": f"Cluster {n.cluster}", "color": "#6366f1"},
                )
                # Compute centroid from neighbors in this cluster
                cluster_neighbors = [nb for nb in neighbors if nb.cluster == n.cluster]
                if cluster_neighbors:
                    centroid_x = sum(nb.coordinates[0] for nb in cluster_neighbors) / len(
                        cluster_neighbors
                    )
                    centroid_y = sum(nb.coordinates[1] for nb in cluster_neighbors) / len(
                        cluster_neighbors
                    )
                else:
                    centroid_x, centroid_y = 0.5, 0.5

                clusters.append(
                    {
                        "id": cluster_info["id"],
                        "label": cluster_info.get("label", f"Cluster {n.cluster}"),
                        "color": cluster_info.get("color", "#6366f1"),
                        "centroid": {"x": centroid_x, "y": centroid_y},
                    }
                )

        # Sort clusters by id
        clusters.sort(key=lambda c: c["id"])

        return SearchResult(
            word=word,
            normalized_word=normalized,
            sense=None,  # FAISS doesn't handle sense disambiguation
            available_senses=[SenseInfo(f"{normalized}|WORD", normalized, 100)],
            neighbors=neighbors[:limit],
            clusters=clusters,
        )

    def has_word(self, word: str) -> bool:
        return word.lower().strip() in self.word2idx

    def find_similar_word(self, word: str) -> str | None:
        """Find similar word using Levenshtein distance."""
        from Levenshtein import distance as levenshtein_distance

        normalized = word.lower().strip()
        closest = None
        min_dist = float("inf")

        # Only check first 10000 words for performance
        for vocab_word in self.words[:10000]:
            dist = levenshtein_distance(normalized, vocab_word)
            if dist <= 2 and dist < min_dist:
                min_dist = dist
                closest = vocab_word

        return closest

    def get_suggestions(self, limit: int = 3) -> list[str]:
        return self.words[:limit]

    def get_available_words(self) -> list[str]:
        return self.words[:1000]  # Return subset for API response


def build_index_from_glove(
    glove_path: str,
    output_dir: str,
    max_words: int | None = None,
):
    """Build FAISS index from GloVe embeddings file.

    Args:
        glove_path: Path to GloVe text file (e.g., glove.6B.300d.txt)
        output_dir: Directory to save processed files
        max_words: Maximum number of words to load (None for all)
    """
    import json
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading GloVe embeddings from {glove_path}...")

    words = []
    embeddings = []

    with open(glove_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_words and i >= max_words:
                break

            parts = line.rstrip().split(" ")
            word = parts[0]
            vector = [float(x) for x in parts[1:]]

            words.append(word)
            embeddings.append(vector)

            if (i + 1) % 100000 == 0:
                print(f"  Loaded {i + 1} words...")

    print(f"Loaded {len(words)} words")

    # Convert to numpy
    embeddings_np = np.array(embeddings, dtype=np.float32)

    # Save words
    with open(output_dir / "words.json", "w") as f:
        json.dump(words, f)

    # Save word to index mapping
    word2idx = {w: i for i, w in enumerate(words)}
    with open(output_dir / "word2idx.json", "w") as f:
        json.dump(word2idx, f)

    # Save embeddings
    np.save(output_dir / "embeddings.npy", embeddings_np)

    print("Computing UMAP coordinates...")
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
        coordinates = reducer.fit_transform(embeddings_np)
        # Normalize to [0, 1]
        coordinates = (coordinates - coordinates.min(axis=0)) / (
            coordinates.max(axis=0) - coordinates.min(axis=0)
        )
        np.save(output_dir / "coordinates.npy", coordinates.astype(np.float32))
    else:
        print("  UMAP not available, skipping coordinate generation")

    print("Building FAISS index...")
    # The index will be built when FAISSEmbeddingProvider is initialized

    print(f"Done! Data saved to {output_dir}")
    print("\nTo use these embeddings:")
    print("  1. Set EMBEDDING_PROVIDER=faiss")
    print(f"  2. Set EMBEDDING_DATA_DIR={output_dir}")
