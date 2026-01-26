"""Datamuse API provider for semantic similarity search.

This provider uses the Datamuse API to find words with similar meanings,
providing true semantic similarity rather than distributional similarity.
"""

import math
import random
from typing import Any

import httpx
import numpy as np

from .base import EmbeddingProvider, SearchResult, SenseInfo, WordResult

# Try to import sklearn for k-means clustering
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KMeans = None
    silhouette_score = None

# Cluster color palette - supports up to 12 clusters
CLUSTER_COLORS = [
    "#6366f1",  # indigo
    "#ec4899",  # pink
    "#14b8a6",  # teal
    "#f59e0b",  # amber
    "#8b5cf6",  # violet
    "#10b981",  # emerald
    "#f43f5e",  # rose
    "#0ea5e9",  # sky
    "#84cc16",  # lime
    "#f97316",  # orange
    "#a855f7",  # purple
    "#06b6d4",  # cyan
]

# Part of speech labels
POS_LABELS = {
    "n": "noun",
    "v": "verb",
    "adj": "adjective",
    "adv": "adverb",
    "u": "unknown",
}


class DatamuseProvider(EmbeddingProvider):
    """Embedding provider using Datamuse API for semantic similarity.

    Uses the 'ml' (means like) parameter for true semantic similarity,
    finding words with similar meanings rather than words that appear
    in similar contexts.

    Optionally uses GloVe embeddings for semantic clustering (grouping
    words by meaning rather than just part of speech).
    """

    def __init__(self):
        self.api_base = "https://api.datamuse.com/words"
        # Cache for spell checking suggestions
        self._suggestion_cache: dict[str, list[str]] = {}
        # Optional embeddings for semantic clustering
        self._embeddings: np.ndarray | None = None
        self._word2idx: dict[str, int] | None = None

    def set_embeddings(self, embeddings: np.ndarray, word2idx: dict[str, int]) -> None:
        """Set GloVe embeddings for semantic clustering.

        Args:
            embeddings: (N, D) numpy array of word embeddings
            word2idx: Dict mapping word -> index in embeddings array
        """
        self._embeddings = embeddings
        self._word2idx = word2idx
        print(f"DatamuseProvider: Enabled semantic clustering with {len(word2idx)} word vectors")

    def _fetch_from_api(self, params: dict[str, str]) -> list[dict[str, Any]]:
        """Fetch results from Datamuse API."""
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(self.api_base, params=params)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"Datamuse API error: {e}")
            return []

    def _get_primary_pos(self, tags: list[str]) -> str:
        """Extract primary part of speech from tags."""
        for tag in tags:
            if tag in POS_LABELS:
                return tag
        return "u"

    def _avoid_collisions(
        self,
        coordinates: list[tuple[float, float]],
        min_distance: float = 0.04,
        iterations: int = 20
    ) -> list[tuple[float, float]]:
        """Push overlapping points apart to avoid text collisions.

        Uses spatial hashing for O(n) average case instead of O(n²).

        Args:
            coordinates: Initial coordinate list
            min_distance: Minimum distance between points
            iterations: Number of repulsion iterations

        Returns:
            Adjusted coordinates with reduced overlap
        """
        if len(coordinates) < 2:
            return coordinates

        coords = [list(c) for c in coordinates]
        n = len(coords)

        # Push points away from center (0.5, 0.5) where query word sits
        center_min_dist = 0.04

        # Cell size for spatial hashing (slightly larger than min_distance)
        cell_size = min_distance * 1.5

        for _ in range(iterations):
            # Push away from center
            for i in range(n):
                dx = coords[i][0] - 0.5
                dy = coords[i][1] - 0.5
                dist_from_center = math.sqrt(dx * dx + dy * dy)

                if dist_from_center < center_min_dist and dist_from_center > 0.001:
                    push = (center_min_dist - dist_from_center) * 0.5
                    coords[i][0] += (dx / dist_from_center) * push
                    coords[i][1] += (dy / dist_from_center) * push

            # Build spatial hash grid
            grid: dict[tuple[int, int], list[int]] = {}
            for i in range(n):
                cx = int(coords[i][0] / cell_size)
                cy = int(coords[i][1] / cell_size)
                key = (cx, cy)
                if key not in grid:
                    grid[key] = []
                grid[key].append(i)

            # Push points apart using spatial hash (only check nearby cells)
            for i in range(n):
                cx = int(coords[i][0] / cell_size)
                cy = int(coords[i][1] / cell_size)

                # Check current cell and 8 neighbors
                for dcx in (-1, 0, 1):
                    for dcy in (-1, 0, 1):
                        neighbor_key = (cx + dcx, cy + dcy)
                        if neighbor_key not in grid:
                            continue

                        for j in grid[neighbor_key]:
                            if j <= i:  # Avoid duplicate pairs
                                continue

                            dx = coords[j][0] - coords[i][0]
                            dy = coords[j][1] - coords[i][1]
                            dist = math.sqrt(dx * dx + dy * dy)

                            if dist < min_distance and dist > 0.001:
                                overlap = (min_distance - dist) * 0.3
                                dx_norm = dx / dist
                                dy_norm = dy / dist

                                coords[i][0] -= dx_norm * overlap
                                coords[i][1] -= dy_norm * overlap
                                coords[j][0] += dx_norm * overlap
                                coords[j][1] += dy_norm * overlap

        # Clamp to valid range (allow full 0-1 range)
        return [
            (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))
            for x, y in coords
        ]

    def _compute_coordinates(
        self,
        words: list[dict[str, Any]],
        clusters: list[int],
        query_word: str,
        layout: str = "sectors"
    ) -> list[tuple[float, float]]:
        """Generate 2D coordinates for visualization.

        Args:
            words: List of word data from Datamuse
            clusters: Cluster assignments for each word
            query_word: The search term (for consistent random seed)
            layout: Layout algorithm - "sectors", "rings", "force", or "grid"
        """
        if not words:
            return []

        random.seed(hash(query_word))

        if layout == "sectors":
            coords = self._layout_sectors(words, clusters)
        elif layout == "rings":
            coords = self._layout_rings(words, clusters)
        elif layout == "force":
            coords = self._layout_force(words, clusters)
        elif layout == "grid":
            coords = self._layout_grid(words, clusters)
        else:
            coords = self._layout_sectors(words, clusters)

        # Apply collision avoidance to reduce text overlaps
        return self._avoid_collisions(coords)

    def _layout_sectors(
        self,
        words: list[dict[str, Any]],
        clusters: list[int]
    ) -> list[tuple[float, float]]:
        """Cluster sectors layout - pie slices by cluster, rank = distance."""
        coordinates = []
        n = len(words)
        if n == 0:
            return []

        # Count words per cluster to allocate angular space
        cluster_counts: dict[int, int] = {}
        for c in clusters:
            cluster_counts[c] = cluster_counts.get(c, 0) + 1

        # Assign angular ranges to each cluster (evenly distributed around circle)
        unique_clusters = sorted(cluster_counts.keys())
        num_clusters = len(unique_clusters)
        cluster_angles: dict[int, tuple[float, float]] = {}

        for idx, cluster_id in enumerate(unique_clusters):
            # Each cluster gets an equal slice of the pie
            start_angle = (idx / num_clusters) * 2 * math.pi
            end_angle = ((idx + 1) / num_clusters) * 2 * math.pi * 0.85  # 85% to leave gaps
            cluster_angles[cluster_id] = (start_angle, start_angle + (end_angle - start_angle))

        # Track position within each cluster
        cluster_positions: dict[int, int] = {c: 0 for c in unique_clusters}

        for i, word_data in enumerate(words):
            cluster = clusters[i]
            tags = word_data.get("tags", [])

            # Use rank for distance (earlier in list = closer to center)
            # Synonyms (marked with "syn" tag) get placed closest
            is_synonym = "syn" in tags
            if is_synonym:
                # Synonyms in inner ring
                distance = 0.08 + (cluster_positions[cluster] / max(cluster_counts[cluster], 1)) * 0.15
            else:
                # Other words spread outward based on position
                distance = 0.25 + (i / n) * 0.20

            # Get angle within cluster's sector
            start_angle, end_angle = cluster_angles[cluster]
            count_in_cluster = cluster_counts[cluster]
            pos_in_cluster = cluster_positions[cluster]
            cluster_positions[cluster] += 1

            if count_in_cluster > 1:
                # Spread within sector
                t = pos_in_cluster / (count_in_cluster - 1)
                angle = start_angle + t * (end_angle - start_angle)
            else:
                angle = (start_angle + end_angle) / 2

            # Add jitter to prevent overlaps
            angle += (random.random() - 0.5) * 0.3
            distance += (random.random() - 0.5) * 0.06

            x = 0.5 + distance * math.cos(angle)
            y = 0.5 + distance * math.sin(angle)

            coordinates.append((max(0.05, min(0.95, x)), max(0.05, min(0.95, y))))

        return coordinates

    def _layout_rings(
        self,
        words: list[dict[str, Any]],
        clusters: list[int]
    ) -> list[tuple[float, float]]:
        """Concentric rings by rank, colored by cluster."""
        n = len(words)
        if n == 0:
            return []

        coordinates: list[tuple[float, float]] = [(0.5, 0.5)] * n

        # Group into rings based on rank (position in list)
        # Synonyms in inner ring, then expand outward
        num_rings = 4
        words_per_ring = max(1, n // num_rings)

        for i, word_data in enumerate(words):
            tags = word_data.get("tags", [])
            is_synonym = "syn" in tags

            # Determine ring based on position
            if is_synonym:
                ring = 0  # Synonyms in innermost ring
            else:
                ring = min(num_rings - 1, 1 + (i // words_per_ring))

            # Ring radius
            radius = 0.10 + ring * 0.10

            # Angle based on cluster and position within ring
            cluster = clusters[i]
            # Offset angle by cluster to group same-cluster words
            base_angle = (cluster / 5) * 2 * math.pi
            # Add position-based offset
            position_offset = (i % words_per_ring) / max(words_per_ring, 1) * (2 * math.pi / 5)
            angle = base_angle + position_offset

            # Add jitter
            angle += (random.random() - 0.5) * 0.4
            radius += (random.random() - 0.5) * 0.04

            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)

            coordinates[i] = (max(0.05, min(0.95, x)), max(0.05, min(0.95, y)))

        return coordinates

    def _layout_force(
        self,
        words: list[dict[str, Any]],
        clusters: list[int]
    ) -> list[tuple[float, float]]:
        """Cluster cloud layout - each cluster in its own region with organic spread."""
        n = len(words)
        if n == 0:
            return []

        max_score = words[0].get("score", 1) if words else 1

        # Assign each cluster a center position (spread around the space)
        unique_clusters = sorted(set(clusters))
        num_clusters = len(unique_clusters)

        cluster_centers = {}
        if num_clusters == 1:
            cluster_centers[unique_clusters[0]] = (0.5, 0.5)
        else:
            for idx, c in enumerate(unique_clusters):
                angle = (idx / num_clusters) * 2 * math.pi - math.pi / 2  # Start from top
                radius = 0.28
                cluster_centers[c] = (
                    0.5 + radius * math.cos(angle),
                    0.5 + radius * math.sin(angle)
                )

        # Group words by cluster and sort by score within each
        cluster_words: dict[int, list[tuple[int, float]]] = {c: [] for c in unique_clusters}
        for i, (word_data, cluster) in enumerate(zip(words, clusters)):
            score = word_data.get("score", 0)
            cluster_words[cluster].append((i, score / max_score if max_score > 0 else 0))

        # Sort each cluster by similarity (highest first)
        for c in cluster_words:
            cluster_words[c].sort(key=lambda x: -x[1])

        # Position words within each cluster
        coordinates: list[tuple[float, float]] = [(0.5, 0.5)] * n

        for cluster_id, words_in_cluster in cluster_words.items():
            cx, cy = cluster_centers[cluster_id]
            count = len(words_in_cluster)

            for idx, (word_i, similarity) in enumerate(words_in_cluster):
                if count == 1:
                    x, y = cx, cy
                else:
                    # Spiral out from cluster center, most similar closest
                    t = idx / count
                    angle = t * 4 * math.pi + random.random() * 0.5  # Spiral
                    radius = 0.03 + t * 0.15  # Expand outward

                    x = cx + radius * math.cos(angle)
                    y = cy + radius * math.sin(angle)

                    # Add jitter
                    x += (random.random() - 0.5) * 0.04
                    y += (random.random() - 0.5) * 0.04

                coordinates[word_i] = (
                    max(0.05, min(0.95, x)),
                    max(0.05, min(0.95, y))
                )

        return coordinates

    def _layout_grid(
        self,
        words: list[dict[str, Any]],
        clusters: list[int]
    ) -> list[tuple[float, float]]:
        """Grid layout - each cluster in a distinct region."""
        coordinates: list[tuple[float, float]] = [(0.5, 0.5)] * len(words)
        max_score = words[0].get("score", 1) if words else 1

        # Define cluster regions (center_x, center_y, width, height)
        cluster_regions = {
            0: (0.5, 0.25, 0.8, 0.35),   # Synonyms - top center
            1: (0.25, 0.7, 0.4, 0.45),   # Nouns - bottom left
            2: (0.75, 0.7, 0.4, 0.45),   # Adjectives - bottom right
            3: (0.5, 0.85, 0.6, 0.25),   # Verbs - bottom center
            4: (0.5, 0.5, 0.3, 0.3),     # Other - middle
        }

        # Group words by cluster
        cluster_words: dict[int, list[int]] = {}
        for i, c in enumerate(clusters):
            if c not in cluster_words:
                cluster_words[c] = []
            cluster_words[c].append(i)

        # Place words in each cluster region
        for cluster_id, word_indices in cluster_words.items():
            region = cluster_regions.get(cluster_id, (0.5, 0.5, 0.3, 0.3))
            cx, cy, w, h = region

            # Sort by similarity (most similar first, placed more prominently)
            word_indices.sort(key=lambda i: -words[i].get("score", 0))

            # Arrange in a grid within the region
            count = len(word_indices)
            cols = max(1, int(math.ceil(math.sqrt(count * w / h))))
            rows = max(1, int(math.ceil(count / cols)))

            for idx, word_i in enumerate(word_indices):
                row = idx // cols
                col = idx % cols

                # Position within region
                x = cx - w/2 + (col + 0.5) * (w / cols)
                y = cy - h/2 + (row + 0.5) * (h / rows)

                # Add jitter
                x += (random.random() - 0.5) * 0.03
                y += (random.random() - 0.5) * 0.03

                coordinates[word_i] = (max(0.02, min(0.98, x)), max(0.02, min(0.98, y)))

        return coordinates

    def _find_optimal_clusters(
        self,
        X: np.ndarray,
        min_k: int = 3,
        max_k: int = 12,
    ) -> int:
        """Determine number of clusters based on word count.

        For thesaurus exploration, silhouette scores are often too low to be
        meaningful (word embeddings form continuous semantic spaces, not
        distinct clusters). Instead, we use a word-count-based heuristic
        that provides good granularity for exploration.

        Args:
            X: Normalized feature matrix
            min_k: Minimum number of clusters
            max_k: Maximum number of clusters

        Returns:
            Number of clusters to use
        """
        n_samples = X.shape[0]

        # Heuristic: roughly 15-20 words per cluster, bounded by min/max
        # This gives: 50 words → 3 clusters, 100 → 6, 150 → 8-10, 200 → 10-12
        target_per_cluster = 18
        k = max(min_k, min(max_k, n_samples // target_per_cluster))

        # Ensure we have at least 5 words per cluster
        k = min(k, n_samples // 5)
        k = max(min_k, k)

        return k

    def _cluster_by_embeddings(
        self,
        words: list[dict[str, Any]],
        max_clusters: int = 12,
    ) -> tuple[list[int], list[str]]:
        """Cluster words by semantic similarity using GloVe embeddings.

        Uses silhouette score to find the optimal number of clusters,
        allowing more clusters for polysemous words with many meanings.

        Args:
            words: List of word data from Datamuse
            max_clusters: Maximum number of clusters to consider

        Returns:
            Tuple of (cluster assignments, cluster labels)
        """
        if not SKLEARN_AVAILABLE or self._embeddings is None or self._word2idx is None:
            return None, None

        # Collect embeddings for words we have vectors for
        word_texts = [w.get("word", "") for w in words]
        valid_indices = []
        valid_vectors = []

        for i, word in enumerate(word_texts):
            word_lower = word.lower()
            if word_lower in self._word2idx:
                idx = self._word2idx[word_lower]
                valid_indices.append(i)
                valid_vectors.append(self._embeddings[idx])

        # Need at least 10 words with embeddings for meaningful clustering
        if len(valid_vectors) < 10:
            return None, None

        # Stack vectors and run k-means
        X = np.vstack(valid_vectors).astype(np.float32)

        # Normalize vectors for cosine-like clustering
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        X_normalized = X / norms

        # Find optimal number of clusters using silhouette score
        optimal_k = self._find_optimal_clusters(X_normalized, min_k=2, max_k=max_clusters)

        kmeans = KMeans(
            n_clusters=optimal_k,
            random_state=42,
            n_init=10,
        )
        cluster_labels = kmeans.fit_predict(X_normalized)

        # Create full assignment list (default to -1 for words without embeddings)
        assignments = [-1] * len(words)
        for i, cluster_id in zip(valid_indices, cluster_labels):
            assignments[i] = int(cluster_id)

        # Assign words without embeddings to nearest cluster based on POS
        for i, assignment in enumerate(assignments):
            if assignment == -1:
                # Fall back to POS-based assignment, but map to valid cluster range
                pos_cluster = self._get_pos_cluster(words[i])
                assignments[i] = pos_cluster % optimal_k

        # Generate cluster labels based on most common words in each cluster
        cluster_label_names = []
        for c in range(optimal_k):
            cluster_words = [word_texts[i] for i, a in enumerate(assignments) if a == c][:3]
            if cluster_words:
                cluster_label_names.append(", ".join(cluster_words[:2]) + "...")
            else:
                cluster_label_names.append(f"group {c + 1}")

        print(f"Optimal clusters: {optimal_k} (from {len(valid_vectors)} words with embeddings)")

        return assignments, cluster_label_names

    def _get_pos_cluster(self, word_data: dict[str, Any]) -> int:
        """Get cluster ID based on part of speech (fallback)."""
        tags = word_data.get("tags", [])
        pos_tags = {"n", "v", "adj", "adv"}

        for tag in tags:
            if tag in pos_tags:
                if tag == "n":
                    return 1
                elif tag == "adj":
                    return 2
                elif tag == "v":
                    return 3
        return 4  # Other

    def _assign_clusters(
        self,
        words: list[dict[str, Any]],
        precomputed_assignments: list[int] | None = None,
        precomputed_labels: list[str] | None = None,
    ) -> tuple[list[int], list[str] | None]:
        """Assign cluster IDs, using precomputed assignments if available.

        Args:
            words: List of word data from Datamuse
            precomputed_assignments: Optional pre-computed cluster assignments from _smart_sample
            precomputed_labels: Optional pre-computed cluster labels from _smart_sample

        Returns:
            Tuple of (cluster assignments, optional cluster labels)
        """
        # Use precomputed assignments if available (from _smart_sample)
        if precomputed_assignments is not None:
            return precomputed_assignments, precomputed_labels

        # Try semantic clustering
        if self._embeddings is not None:
            assignments, labels = self._cluster_by_embeddings(words)
            if assignments is not None:
                return assignments, labels

        # Fall back to POS-based clustering
        clusters = []
        pos_tags = {"n", "v", "adj", "adv"}

        for word_data in words:
            tags = word_data.get("tags", [])

            # Check if it's a synonym first
            if "syn" in tags:
                clusters.append(0)  # Synonyms cluster
                continue

            # Find the first POS tag (Datamuse orders by primary usage)
            first_pos = None
            for tag in tags:
                if tag in pos_tags:
                    first_pos = tag
                    break

            if first_pos == "n":
                clusters.append(1)  # Nouns
            elif first_pos == "adj":
                clusters.append(2)  # Adjectives
            elif first_pos == "v":
                clusters.append(3)  # Verbs
            else:
                clusters.append(4)  # Other (adverbs, unknown)

        return clusters, None  # No custom labels for POS-based clustering

    def _get_frequency_tier(self, score: int, max_score: int) -> str:
        """Determine frequency tier based on score."""
        if max_score == 0:
            return "common"

        ratio = score / max_score
        if ratio > 0.5:
            return "common"
        elif ratio > 0.2:
            return "uncommon"
        else:
            return "rare"

    def _filter_by_frequency(
        self,
        results: list[dict[str, Any]],
        include_rare: bool = False,
        common_threshold: float = 0.5,
        limit: int = 100,
        rare_bonus: int = 50
    ) -> list[dict[str, Any]]:
        """Filter results by frequency.

        When include_rare is False, only return common words (up to limit).
        When include_rare is True, return common words (up to limit) + rare words (up to rare_bonus extra).
        This ensures rare mode always adds value even for words with many common synonyms.

        Args:
            results: Raw results from Datamuse API
            include_rare: Whether to include rare words
            common_threshold: Frequency threshold for "common" words
            limit: Maximum common words to return
            rare_bonus: Extra rare words to add when include_rare is True

        Returns:
            Filtered list - common words, optionally followed by rare words
        """
        common_words = []
        rare_words = []

        for word_data in results:
            tags = word_data.get("tags", [])

            # Check frequency
            freq = 0.0
            for tag in tags:
                if tag.startswith("f:"):
                    try:
                        freq = float(tag[2:])
                    except ValueError:
                        pass
                    break

            if freq >= common_threshold:
                common_words.append(word_data)
            else:
                rare_words.append(word_data)

        if include_rare:
            # Return common words (up to limit) + rare words (up to rare_bonus extra)
            return common_words[:limit] + rare_words[:rare_bonus]
        else:
            # Return only common words (up to limit)
            return common_words[:limit]

    def _smart_sample(
        self,
        results: list[dict[str, Any]],
        limit: int = 150,
        min_per_cluster: int = 8,
        max_clusters: int = 12,
    ) -> tuple[list[dict[str, Any]], list[int] | None, list[str] | None]:
        """Sample results to ensure diversity across semantic clusters.

        Instead of just taking the top N results (which may all be from the
        most common sense), this samples proportionally from each semantic
        cluster while ensuring minimum representation.

        Args:
            results: All candidate results
            limit: Target number of results to return
            min_per_cluster: Minimum words per cluster (if available)
            max_clusters: Maximum number of clusters to create

        Returns:
            Tuple of (sampled_results, cluster_assignments, cluster_labels)
            Cluster assignments and labels are returned to avoid re-clustering later.
        """
        if len(results) <= limit:
            return results, None, None

        # Try semantic clustering
        if not SKLEARN_AVAILABLE or self._embeddings is None or self._word2idx is None:
            # Fall back to simple truncation
            return results[:limit], None, None

        # Get embeddings for clustering
        word_texts = [w.get("word", "").lower() for w in results]
        valid_indices = []
        valid_vectors = []

        for i, word in enumerate(word_texts):
            if word in self._word2idx:
                idx = self._word2idx[word]
                valid_indices.append(i)
                valid_vectors.append(self._embeddings[idx])

        # Need enough words with embeddings to cluster
        if len(valid_vectors) < 20:
            return results[:limit], None, None

        # Cluster the words
        X = np.vstack(valid_vectors).astype(np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X_normalized = X / norms

        # Find optimal number of clusters for sampling
        actual_n_clusters = self._find_optimal_clusters(X_normalized, min_k=2, max_k=max_clusters)

        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
        cluster_labels_arr = kmeans.fit_predict(X_normalized)

        # Build cluster -> results mapping
        # Words without embeddings go to cluster -1
        cluster_assignments = [-1] * len(results)
        for i, cluster_id in zip(valid_indices, cluster_labels_arr):
            cluster_assignments[i] = int(cluster_id)

        clusters: dict[int, list[int]] = {}
        for i, c in enumerate(cluster_assignments):
            if c not in clusters:
                clusters[c] = []
            clusters[c].append(i)

        # Sample from each cluster
        sampled_indices: list[int] = []
        remaining = limit

        # Sort clusters by size (largest first) but process all
        sorted_clusters = sorted(
            [(c, indices) for c, indices in clusters.items() if c >= 0],
            key=lambda x: -len(x[1])
        )

        # First pass: ensure minimum per cluster
        for cluster_id, indices in sorted_clusters:
            take = min(min_per_cluster, len(indices), remaining)
            sampled_indices.extend(indices[:take])
            remaining -= take
            if remaining <= 0:
                break

        # Second pass: fill remaining proportionally from what's left
        if remaining > 0:
            total_available = sum(
                len(indices) - min(min_per_cluster, len(indices))
                for _, indices in sorted_clusters
            )
            if total_available > 0:
                for cluster_id, indices in sorted_clusters:
                    already_taken = min(min_per_cluster, len(indices))
                    available = indices[already_taken:]
                    if not available:
                        continue
                    proportion = len(available) / total_available
                    take = min(max(1, int(remaining * proportion)), len(available))
                    sampled_indices.extend(available[:take])

        # Include some words without embeddings if we have room
        no_embedding = clusters.get(-1, [])
        if len(sampled_indices) < limit and no_embedding:
            take = min(limit - len(sampled_indices), len(no_embedding))
            sampled_indices.extend(no_embedding[:take])

        # Sort by original order to preserve Datamuse relevance ranking within clusters
        sampled_indices = sorted(set(sampled_indices))[:limit]

        # Build mapping from old index to new index for cluster assignments
        sampled_results = [results[i] for i in sampled_indices]
        sampled_assignments = [cluster_assignments[i] for i in sampled_indices]

        # Remap cluster IDs to be contiguous (0, 1, 2, ...)
        unique_clusters = sorted(set(c for c in sampled_assignments if c >= 0))
        cluster_remap = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
        sampled_assignments = [cluster_remap.get(c, 0) for c in sampled_assignments]

        # Generate cluster labels
        cluster_label_names = []
        for new_id in range(len(unique_clusters)):
            cluster_words = [
                sampled_results[i].get("word", "")
                for i, a in enumerate(sampled_assignments) if a == new_id
            ][:3]
            if cluster_words:
                cluster_label_names.append(", ".join(cluster_words[:2]) + "...")
            else:
                cluster_label_names.append(f"group {new_id + 1}")

        print(f"Optimal clusters: {len(unique_clusters)} (from {len(valid_vectors)} words with embeddings)")

        return sampled_results, sampled_assignments, cluster_label_names

    def search(
        self,
        word: str,
        sense: str | None = None,
        limit: int = 100,
        layout: str = "sectors",
        include_rare: bool = False,
    ) -> SearchResult | None:
        normalized = word.lower().strip()

        if not normalized:
            return None

        # Fetch more candidates than needed to enable smart sampling across senses
        # This ensures we get diverse representation even for polysemous words
        # Datamuse can return up to 1000 results - fetch more for better clustering
        fetch_limit = max(limit * 4, 500)

        # First, fetch explicit synonyms (these are always good)
        synonyms = self._fetch_from_api({
            "rel_syn": normalized,
            "max": str(min(fetch_limit, 300)),
            "md": "fp",
        })

        # Mark all synonym results with "syn" tag
        synonym_words = set()
        for syn in synonyms:
            syn_word = syn.get("word", "")
            synonym_words.add(syn_word)
            if "tags" not in syn:
                syn["tags"] = []
            syn["tags"].append("syn")

        # Also fetch "means like" results for broader coverage
        # Use higher limit to capture more senses of polysemous words
        ml_results = self._fetch_from_api({
            "ml": normalized,
            "max": str(min(fetch_limit, 1000)),
            "md": "fp",
        })

        # Combine: synonyms first, then ml results (avoiding duplicates)
        results = list(synonyms)
        seen_words = set(synonym_words)
        for ml_word in ml_results:
            word_text = ml_word.get("word", "")
            if word_text not in seen_words:
                results.append(ml_word)
                seen_words.add(word_text)

        if not results:
            return None

        # Filter out multi-word phrases for cleaner visualization
        results = [r for r in results if " " not in r.get("word", "")]

        # Filter by frequency first (get more than limit to allow smart sampling)
        frequency_limit = limit * 3 if self._embeddings is not None else limit
        results = self._filter_by_frequency(
            results, include_rare=include_rare, limit=frequency_limit, rare_bonus=150
        )

        # Smart sample to ensure diversity across semantic clusters
        # Returns precomputed cluster info to avoid duplicate clustering
        results, precomputed_assignments, precomputed_labels = self._smart_sample(
            results, limit=limit, max_clusters=12
        )

        if not results:
            return None

        # Compute visualization data (reuse cluster info from sampling if available)
        cluster_assignments, custom_cluster_labels = self._assign_clusters(
            results, precomputed_assignments, precomputed_labels
        )
        coordinates = self._compute_coordinates(results, cluster_assignments, normalized, layout)
        # Find the true max score across all results for proper normalization
        max_score = max((r.get("score", 0) for r in results), default=1)
        if max_score == 0:
            max_score = 1

        # Build neighbors list
        neighbors = []

        # Add the query word itself at center
        neighbors.append(
            WordResult(
                word=normalized,
                similarity=1.0,
                coordinates=(0.5, 0.5),
                frequency="common",
                cluster=0,
            )
        )

        for i, result in enumerate(results):
            result_word = result.get("word", "")
            score = result.get("score", 0)

            # Normalize similarity score to 0-1 range
            similarity = score / max_score if max_score > 0 else 0

            neighbors.append(
                WordResult(
                    word=result_word,
                    similarity=round(similarity, 3),
                    coordinates=coordinates[i] if i < len(coordinates) else (0.5, 0.5),
                    frequency=self._get_frequency_tier(score, max_score),
                    cluster=cluster_assignments[i] if i < len(cluster_assignments) else 0,
                )
            )

        # Build cluster metadata
        seen_clusters = set(cluster_assignments)

        # Use custom labels from semantic clustering, or fall back to POS labels
        if custom_cluster_labels:
            cluster_labels = {i: label for i, label in enumerate(custom_cluster_labels)}
        else:
            cluster_labels = {
                0: "synonyms",
                1: "related nouns",
                2: "related adjectives",
                3: "related verbs",
                4: "other related",
            }

        clusters = []
        for cluster_id in sorted(seen_clusters):
            # Compute centroid from neighbors in this cluster
            cluster_neighbors = [n for n in neighbors if n.cluster == cluster_id]
            if cluster_neighbors:
                centroid_x = sum(n.coordinates[0] for n in cluster_neighbors) / len(cluster_neighbors)
                centroid_y = sum(n.coordinates[1] for n in cluster_neighbors) / len(cluster_neighbors)
            else:
                centroid_x, centroid_y = 0.5, 0.5

            clusters.append({
                "id": cluster_id,
                "label": cluster_labels.get(cluster_id, f"group {cluster_id + 1}"),
                "color": CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
                "centroid": {"x": centroid_x, "y": centroid_y},
            })

        return SearchResult(
            word=word,
            normalized_word=normalized,
            sense=None,
            available_senses=[SenseInfo(f"{normalized}|SEMANTIC", f"{normalized} (semantic)", 100)],
            neighbors=neighbors,
            clusters=clusters,
        )

    def expand_cluster(
        self,
        word: str,
        anchor_words: list[str],
        limit: int = 50,
        exclude_words: list[str] | None = None,
    ) -> SearchResult | None:
        """Expand a semantic cluster by fetching more words similar to anchor words.

        Args:
            word: The original query word
            anchor_words: Representative words from the cluster to expand
            limit: Number of additional words to fetch
            exclude_words: Words already shown (to avoid duplicates)

        Returns:
            SearchResult with additional words from the same semantic cluster
        """
        if not anchor_words:
            return None

        normalized = word.lower().strip()
        exclude_set = set(w.lower() for w in (exclude_words or []))
        exclude_set.add(normalized)

        # Fetch results similar to each anchor word
        all_candidates: list[dict[str, Any]] = []
        seen_words: set[str] = set()

        for anchor in anchor_words[:5]:  # Limit anchors to avoid too many API calls
            # Fetch words similar to this anchor
            results = self._fetch_from_api({
                "ml": anchor.lower().strip(),
                "max": "100",
                "md": "fp",
            })

            for r in results:
                w = r.get("word", "").lower()
                if w not in seen_words and w not in exclude_set and " " not in w:
                    seen_words.add(w)
                    all_candidates.append(r)

        if not all_candidates:
            return None

        # If we have embeddings, filter to words closest to anchor centroid
        if self._embeddings is not None and self._word2idx is not None:
            # Compute anchor centroid
            anchor_vectors = []
            for anchor in anchor_words:
                anchor_lower = anchor.lower()
                if anchor_lower in self._word2idx:
                    idx = self._word2idx[anchor_lower]
                    anchor_vectors.append(self._embeddings[idx])

            if anchor_vectors:
                centroid = np.mean(anchor_vectors, axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

                # Score candidates by similarity to centroid
                scored_candidates = []
                for candidate in all_candidates:
                    w = candidate.get("word", "").lower()
                    if w in self._word2idx:
                        idx = self._word2idx[w]
                        vec = self._embeddings[idx]
                        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
                        similarity = float(np.dot(centroid, vec_norm))
                        scored_candidates.append((similarity, candidate))

                # Sort by similarity and take top results
                scored_candidates.sort(key=lambda x: -x[0])
                all_candidates = [c for _, c in scored_candidates[:limit * 2]]

        # Take the requested limit
        results = all_candidates[:limit]

        if not results:
            return None

        # Compute visualization - use simple clustering for expanded results
        cluster_assignments, custom_labels = self._assign_clusters(results)
        coordinates = self._compute_coordinates(results, cluster_assignments, normalized, "force")

        max_score = max((r.get("score", 0) for r in results), default=1)
        if max_score == 0:
            max_score = 1

        neighbors = []
        for i, result in enumerate(results):
            result_word = result.get("word", "")
            score = result.get("score", 0)
            similarity = score / max_score if max_score > 0 else 0

            neighbors.append(
                WordResult(
                    word=result_word,
                    similarity=round(similarity, 3),
                    coordinates=coordinates[i] if i < len(coordinates) else (0.5, 0.5),
                    frequency=self._get_frequency_tier(score, max_score),
                    cluster=cluster_assignments[i] if i < len(cluster_assignments) else 0,
                )
            )

        # Single cluster for expanded results
        clusters = [{
            "id": 0,
            "label": f"expanded: {', '.join(anchor_words[:2])}...",
            "color": CLUSTER_COLORS[0],
            "centroid": {"x": 0.5, "y": 0.5},
        }]

        return SearchResult(
            word=word,
            normalized_word=normalized,
            sense=None,
            available_senses=[],
            neighbors=neighbors,
            clusters=clusters,
        )

    def has_word(self, word: str) -> bool:
        """Check if word exists - Datamuse has broad coverage so assume yes."""
        return bool(word and word.strip())

    def find_similar_word(self, word: str) -> str | None:
        """Find similar word using Datamuse spell checking."""
        normalized = word.lower().strip()

        # Use Datamuse's spell suggestion
        results = self._fetch_from_api({
            "sp": normalized,
            "max": "5",
        })

        if results:
            # Return the first suggestion that's different from input
            for result in results:
                suggestion = result.get("word", "")
                if suggestion and suggestion != normalized:
                    return suggestion

        return None

    def get_suggestions(self, limit: int = 3) -> list[str]:
        """Get common word suggestions."""
        return ["happy", "sad", "beautiful", "strong", "love"][:limit]

    def get_available_words(self) -> list[str]:
        """Datamuse has ~550k words - return sample."""
        return ["happy", "sad", "beautiful", "ocean", "mountain", "love", "joy", "fear"]
