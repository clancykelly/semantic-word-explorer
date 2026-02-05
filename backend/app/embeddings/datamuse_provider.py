"""Datamuse API provider for semantic similarity search.

This provider uses the Datamuse API to find words with similar meanings,
providing true semantic similarity rather than distributional similarity.
"""

import math
import random
import time
from collections import Counter
from typing import Any

import httpx
import numpy as np

from .base import EmbeddingProvider, SearchResult, SenseInfo, WordResult
from .moby_thesaurus import get_moby_thesaurus

# Try to import sklearn for clustering
try:
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    AgglomerativeClustering = None
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

# Formal suffixes (Latin/Greek origins, academic register)
FORMAL_SUFFIXES = (
    "tion", "sion", "ment", "ity", "ness", "ance", "ence", "ious", "eous",
    "ical", "ology", "istic", "ism", "ist", "ive", "ary", "ory", "al", "ure",
)

# Casual/informal markers
CASUAL_WORDS = frozenset({
    "gonna", "wanna", "gotta", "kinda", "sorta", "yeah", "yep", "nope",
    "stuff", "thing", "things", "guy", "guys", "cool", "awesome", "okay",
    "ok", "hey", "hi", "bye", "wow", "oops", "ugh", "huh", "yay", "nah",
})


def compute_formality(word: str) -> float:
    """Compute a formality score for a word (0.0 = casual, 1.0 = formal).

    Uses heuristics based on:
    - Word length (longer words tend to be more formal)
    - Syllable count approximation
    - Latin/Greek suffixes (more formal)
    - Known casual words (less formal)
    """
    word_lower = word.lower().strip()

    # Handle phrases - average formality of components
    if " " in word_lower:
        parts = word_lower.split()
        if not parts:
            return 0.5
        return sum(compute_formality(p) for p in parts) / len(parts)

    # Known casual words
    if word_lower in CASUAL_WORDS:
        return 0.1

    # Base score from word length (normalized 3-12 chars to 0.3-0.7)
    length_score = min(1.0, max(0.0, (len(word_lower) - 3) / 9)) * 0.4 + 0.3

    # Syllable approximation (count vowel groups)
    vowels = "aeiouy"
    syllables = 0
    prev_vowel = False
    for char in word_lower:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            syllables += 1
        prev_vowel = is_vowel
    syllables = max(1, syllables)

    # Syllable bonus (more syllables = more formal, normalized 1-5 to 0-0.2)
    syllable_bonus = min(0.2, max(0.0, (syllables - 1) / 4) * 0.2)

    # Formal suffix bonus
    suffix_bonus = 0.0
    for suffix in FORMAL_SUFFIXES:
        if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
            suffix_bonus = 0.15
            break

    # Combine scores
    formality = length_score + syllable_bonus + suffix_bonus

    # Clamp to 0-1
    return min(1.0, max(0.0, formality))


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
        # ConceptNet cache: (word, rel_type) -> (timestamp, results)
        self._conceptnet_cache: dict[tuple[str, str], tuple[float, list[dict[str, Any]]]] = {}
        self._conceptnet_cache_ttl: float = 3600.0  # 1 hour

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

    def _fetch_from_conceptnet(self, word: str) -> list[dict[str, Any]]:
        """Fetch synonyms and related words from ConceptNet.

        Makes two API calls (Synonym and RelatedTo edges), caches results
        for 1 hour, and returns deduplicated candidates tagged with "conceptnet".
        """
        now = time.monotonic()
        results: list[dict[str, Any]] = []
        seen: set[str] = set()

        for rel_type in ("Synonym", "RelatedTo"):
            cache_key = (word, rel_type)

            # Check cache
            if cache_key in self._conceptnet_cache:
                ts, cached = self._conceptnet_cache[cache_key]
                if now - ts < self._conceptnet_cache_ttl:
                    for entry in cached:
                        w = entry.get("word", "").lower()
                        if w not in seen:
                            seen.add(w)
                            results.append(entry)
                    continue

            # Fetch from API
            url = f"https://api.conceptnet.io/query?node=/c/en/{word}&rel=/r/{rel_type}&limit=100"
            fetched: list[dict[str, Any]] = []
            try:
                with httpx.Client(timeout=10.0) as client:
                    response = client.get(url)
                    response.raise_for_status()
                    data = response.json()

                for edge in data.get("edges", []):
                    weight = edge.get("weight", 0)
                    for end_key in ("start", "end"):
                        node = edge.get(end_key, {})
                        lang = node.get("language", "")
                        label = node.get("label", "")
                        if lang != "en" or not label:
                            continue
                        # Convert underscores to spaces
                        label = label.replace("_", " ").strip().lower()
                        if label == word or label in seen:
                            continue
                        seen.add(label)
                        entry = {
                            "word": label,
                            "score": max(1, int(weight * 100)),
                            "tags": ["conceptnet"],
                        }
                        fetched.append(entry)
                        results.append(entry)
            except Exception as e:
                print(f"ConceptNet API error ({rel_type}): {e}")

            # Cache regardless of success (empty list on failure avoids retries)
            self._conceptnet_cache[cache_key] = (now, fetched)

        return results

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

    def _build_distance_matrix(
        self,
        word_texts: list[str],
        glove_weight: float = 0.5,
    ) -> tuple[np.ndarray, int]:
        """Build a combined distance matrix from GloVe cosine + Moby Jaccard similarity.

        For word pairs where both signals exist, blends them. Where only one
        signal exists, uses that signal alone. Where neither exists, uses
        maximum distance.

        Args:
            word_texts: Lowercased word strings
            glove_weight: Weight for GloVe similarity (1 - glove_weight for Moby)

        Returns:
            Tuple of (distance matrix, number of words with at least one signal)
        """
        n = len(word_texts)
        moby = get_moby_thesaurus()

        # Precompute Moby synonym sets
        moby_sets: list[set[str]] = []
        for w in word_texts:
            syns = set(s.lower() for s in moby.get_synonyms(w))
            moby_sets.append(syns)

        # Precompute GloVe vectors (normalized)
        glove_vecs: list[np.ndarray | None] = []
        for w in word_texts:
            if w in self._word2idx:
                v = self._embeddings[self._word2idx[w]].astype(np.float32)
                norm = np.linalg.norm(v)
                glove_vecs.append(v / norm if norm > 0 else v)
            elif " " in w:
                parts = [self._embeddings[self._word2idx[p]] for p in w.split() if p in self._word2idx]
                if parts:
                    v = np.mean(parts, axis=0).astype(np.float32)
                    norm = np.linalg.norm(v)
                    glove_vecs.append(v / norm if norm > 0 else v)
                else:
                    glove_vecs.append(None)
            else:
                glove_vecs.append(None)

        # Build pairwise distance matrix
        dist = np.ones((n, n), dtype=np.float32)
        np.fill_diagonal(dist, 0.0)

        for i in range(n):
            for j in range(i + 1, n):
                has_glove = glove_vecs[i] is not None and glove_vecs[j] is not None
                has_moby = bool(moby_sets[i]) and bool(moby_sets[j])

                if has_glove and has_moby:
                    glove_sim = float(np.dot(glove_vecs[i], glove_vecs[j]))
                    union = moby_sets[i] | moby_sets[j]
                    moby_sim = len(moby_sets[i] & moby_sets[j]) / len(union) if union else 0.0
                    sim = glove_weight * glove_sim + (1 - glove_weight) * moby_sim
                elif has_glove:
                    sim = float(np.dot(glove_vecs[i], glove_vecs[j]))
                elif has_moby:
                    union = moby_sets[i] | moby_sets[j]
                    sim = len(moby_sets[i] & moby_sets[j]) / len(union) if union else 0.0
                else:
                    sim = 0.0

                d = max(0.0, 1.0 - sim)
                dist[i, j] = d
                dist[j, i] = d

        return dist

    def _find_optimal_clusters(
        self,
        dist: np.ndarray,
        min_k: int = 3,
        max_k: int = 10,
    ) -> int:
        """Find optimal cluster count using silhouette score on precomputed distances."""
        n_samples = dist.shape[0]

        heuristic_k = max(min_k, n_samples // 15)
        search_min = max(min_k, heuristic_k - 1)
        search_max = min(max_k, heuristic_k + 2, n_samples // 3)

        if search_min >= search_max:
            return max(min_k, min(search_max, n_samples // 3))

        best_k = search_min
        best_score = -1.0

        for k in range(search_min, search_max + 1):
            ac = AgglomerativeClustering(
                n_clusters=k, metric="precomputed", linkage="average"
            )
            labels = ac.fit_predict(dist)
            score = silhouette_score(dist, labels, metric="precomputed")
            if score > best_score:
                best_score = score
                best_k = k

        return best_k

    def _cluster_by_embeddings(
        self,
        words: list[dict[str, Any]],
        max_clusters: int = 10,
        query_word: str = "",
    ) -> tuple[list[int], list[str]]:
        """Cluster words using blended GloVe + Moby co-synonym similarity.

        Builds a pairwise distance matrix combining GloVe cosine similarity
        with Moby Thesaurus Jaccard overlap, then uses agglomerative clustering.

        Args:
            words: List of word data from Datamuse
            max_clusters: Maximum number of clusters to consider
            query_word: The original query word

        Returns:
            Tuple of (cluster assignments, cluster labels)
        """
        if not SKLEARN_AVAILABLE or self._embeddings is None or self._word2idx is None:
            return None, None

        word_texts = [w.get("word", "").lower() for w in words]
        n_words = len(words)

        if n_words < 5:
            return None, None

        dist = self._build_distance_matrix(word_texts)

        optimal_k = self._find_optimal_clusters(dist, min_k=2, max_k=max_clusters)

        ac = AgglomerativeClustering(
            n_clusters=optimal_k, metric="precomputed", linkage="average"
        )
        assignments = [int(x) for x in ac.fit_predict(dist)]

        # Merge small clusters — scale min_size to dataset size
        adaptive_min_size = max(3, n_words // (optimal_k * 2))
        assignments = self._merge_small_clusters(assignments, dist, min_size=adaptive_min_size)

        n_final = len(set(assignments))

        # Generate descriptive cluster labels
        cluster_label_names = self._generate_cluster_labels(
            words, assignments, n_final
        )

        print(f"Clustering: {n_final} clusters from {n_words} words (hybrid GloVe+Moby)")

        return assignments, cluster_label_names

    def _merge_small_clusters(
        self,
        assignments: list[int],
        dist: np.ndarray,
        min_size: int = 8,
    ) -> list[int]:
        """Merge small clusters into their nearest neighbor cluster.

        Iteratively merges the smallest cluster (below min_size) into the
        closest other cluster by average linkage distance.
        """
        assignments = list(assignments)

        while True:
            cluster_indices: dict[int, list[int]] = {}
            for i, c in enumerate(assignments):
                cluster_indices.setdefault(c, []).append(i)

            # Find smallest cluster below threshold
            small = [(c, idx) for c, idx in cluster_indices.items() if len(idx) < min_size]
            if not small:
                break

            # Merge the smallest one
            small.sort(key=lambda x: len(x[1]))
            merge_c, merge_idx = small[0]

            # Find nearest other cluster by average linkage
            best_target = None
            best_dist = float("inf")
            for other_c, other_idx in cluster_indices.items():
                if other_c == merge_c:
                    continue
                avg_d = float(np.mean([dist[i, j] for i in merge_idx for j in other_idx]))
                if avg_d < best_dist:
                    best_dist = avg_d
                    best_target = other_c

            if best_target is None:
                break

            for i in merge_idx:
                assignments[i] = best_target

        # Remap to contiguous IDs
        unique = sorted(set(assignments))
        remap = {old: new for new, old in enumerate(unique)}
        return [remap[c] for c in assignments]

    def _assign_oov_words(
        self,
        words: list[dict[str, Any]],
        word_texts: list[str],
        assignments: list[int],
        n_clusters: int,
    ) -> None:
        """Assign out-of-vocabulary words to clusters via Moby co-synonym majority vote.

        For each unassigned word, finds its Moby co-synonyms that ARE in the result
        set and assigns it to the majority cluster of those co-synonyms.
        Falls back to the largest cluster if no co-synonyms are found.

        Modifies assignments in place.
        """
        unassigned = [i for i, a in enumerate(assignments) if a == -1]
        if not unassigned:
            return

        moby = get_moby_thesaurus()

        # Build lookup: word -> cluster assignment (for words already assigned)
        word_to_cluster = {}
        for i, a in enumerate(assignments):
            if a >= 0:
                word_to_cluster[word_texts[i].lower()] = a

        # Find largest cluster as fallback
        cluster_sizes = Counter(a for a in assignments if a >= 0)
        fallback_cluster = cluster_sizes.most_common(1)[0][0] if cluster_sizes else 0

        for i in unassigned:
            word_lower = word_texts[i].lower()
            # Get this word's Moby co-synonyms
            oov_synonyms = set(s.lower() for s in moby.get_synonyms(word_lower))

            # Find which clusters those co-synonyms belong to
            co_synonym_clusters = []
            for syn in oov_synonyms:
                if syn in word_to_cluster:
                    co_synonym_clusters.append(word_to_cluster[syn])

            if co_synonym_clusters:
                assignments[i] = Counter(co_synonym_clusters).most_common(1)[0][0]
            else:
                assignments[i] = fallback_cluster

    def _generate_cluster_labels(
        self,
        words: list[dict[str, Any]],
        assignments: list[int],
        n_clusters: int,
    ) -> list[str]:
        """Generate descriptive labels for clusters.

        Args:
            words: List of word data
            assignments: Cluster assignment for each word
            n_clusters: Total number of clusters

        Returns:
            List of cluster labels
        """
        labels = []
        word_texts = [w.get("word", "") for w in words]

        for c in range(n_clusters):
            cluster_words = [word_texts[i] for i, a in enumerate(assignments) if a == c]

            if not cluster_words:
                labels.append(f"group {c + 1}")
                continue

            # Analyze cluster content for better naming
            phrases = [w for w in cluster_words if " " in w]
            single_words = [w for w in cluster_words if " " not in w]

            # Check if it's mostly named bodies of water
            water_indicators = ["ocean", "sea", "gulf", "bay", "strait"]
            geographic_count = sum(1 for w in cluster_words
                                   if any(ind in w.lower() for ind in water_indicators))

            if geographic_count > len(cluster_words) * 0.5 and phrases:
                labels.append("bodies of water")
            elif len(phrases) > len(single_words):
                sample = phrases[:2]
                labels.append(", ".join(sample) + "...")
            else:
                sample = single_words[:2] if single_words else cluster_words[:2]
                labels.append(", ".join(sample) + "...")

        return labels

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
        query_word: str = "",
    ) -> tuple[list[int], list[str] | None]:
        """Assign cluster IDs, using precomputed assignments if available.

        Args:
            words: List of word data from Datamuse
            precomputed_assignments: Optional pre-computed cluster assignments from _smart_sample
            precomputed_labels: Optional pre-computed cluster labels from _smart_sample
            query_word: The original query word for relevance calculation

        Returns:
            Tuple of (cluster assignments, optional cluster labels)
        """
        # Use precomputed assignments if available (from _smart_sample)
        if precomputed_assignments is not None:
            return precomputed_assignments, precomputed_labels

        # Try semantic clustering
        if self._embeddings is not None:
            assignments, labels = self._cluster_by_embeddings(words, query_word=query_word)
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

    def _get_frequency_tier(self, word_data: dict) -> str:
        """Determine frequency tier based on Datamuse frequency tag.

        Uses the f: tag from Datamuse metadata, which represents
        word frequency per million words in a large corpus.
        """
        tags = word_data.get("tags", [])

        # Extract frequency from f: tag
        freq = None
        for tag in tags:
            if tag.startswith("f:"):
                try:
                    freq = float(tag[2:])
                except ValueError:
                    pass
                break

        # Curated sources (Moby, ConceptNet) are treated as common
        is_curated = "moby" in tags or "conceptnet" in tags or "syn" in tags
        if is_curated:
            return "common"

        # No frequency data - assume common
        if freq is None:
            return "common"

        # Datamuse frequency is per million words
        # Common words: freq > 10 (appears 10+ times per million)
        # Uncommon: 1-10
        # Rare: < 1
        if freq > 10:
            return "common"
        elif freq > 1:
            return "uncommon"
        else:
            return "rare"

    def _compute_query_relevance(
        self,
        query_word: str,
        candidates: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], float]]:
        """Compute cosine similarity between each candidate and the query word.

        Args:
            query_word: The query word to compare against
            candidates: List of candidate word data from Datamuse

        Returns:
            List of (word_data, similarity) tuples, sorted by similarity descending.
            Words without embeddings get similarity of 0.0.
        """
        if self._embeddings is None or self._word2idx is None:
            # No embeddings - return all with similarity 1.0 (no filtering)
            return [(c, 1.0) for c in candidates]

        query_lower = query_word.lower()
        if query_lower not in self._word2idx:
            # Query word not in vocabulary - return all with similarity 1.0
            return [(c, 1.0) for c in candidates]

        # Get query vector
        query_idx = self._word2idx[query_lower]
        query_vec = self._embeddings[query_idx]
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)

        # Score each candidate
        scored = []
        for candidate in candidates:
            word = candidate.get("word", "").lower()
            is_synonym = "syn" in candidate.get("tags", [])

            has_embedding = False

            if word in self._word2idx:
                # Single word with embedding
                idx = self._word2idx[word]
                vec = self._embeddings[idx]
                vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
                similarity = float(np.dot(query_norm, vec_norm))
                has_embedding = True
            elif " " in word:
                # Multi-word phrase - compute average embedding of component words
                # EXCLUDE the query word itself to avoid inflated similarity
                # (e.g., "run out" scored only on "out", not "run")
                component_vecs = []
                for component in word.split():
                    component = component.lower()
                    if component == query_lower:
                        continue  # Skip query word in phrase
                    if component in self._word2idx:
                        idx = self._word2idx[component]
                        component_vecs.append(self._embeddings[idx])

                if component_vecs:
                    # Average the component embeddings
                    avg_vec = np.mean(component_vecs, axis=0)
                    avg_norm = avg_vec / (np.linalg.norm(avg_vec) + 1e-8)
                    similarity = float(np.dot(query_norm, avg_norm))
                    has_embedding = True
                elif is_synonym:
                    # Phrase with no known components but marked as synonym - trust it
                    similarity = 0.5
                else:
                    # Unknown phrase - low score
                    similarity = 0.2
            elif is_synonym:
                # Explicit synonym without embedding - trust Datamuse
                similarity = 0.5
            else:
                # No embedding - give benefit of doubt with moderate score
                similarity = 0.25

            candidate["_has_embedding"] = has_embedding
            scored.append((candidate, similarity))

        # Sort by similarity descending
        scored.sort(key=lambda x: -x[1])
        return scored

    def _find_adaptive_threshold(
        self,
        similarities: list[float],
        min_threshold: float = 0.20,
        max_threshold: float = 0.55,
        min_keep: int = 40,
    ) -> float:
        """Find adaptive threshold based on gap detection in similarity scores.

        Looks for the largest gap in sorted similarity scores to find a natural
        cutoff point between related and unrelated words.

        Args:
            similarities: List of similarity scores (should be sorted descending)
            min_threshold: Minimum threshold to use
            max_threshold: Maximum threshold to use
            min_keep: Minimum number of words to keep (won't set threshold higher than this)

        Returns:
            Adaptive threshold value
        """
        if len(similarities) < min_keep:
            return min_threshold

        # Ensure sorted descending
        sorted_sims = sorted(similarities, reverse=True)

        # Look for largest gap in the scores (excluding top min_keep)
        # Start looking after min_keep items to ensure we keep enough words
        best_gap = 0.0
        best_threshold = min_threshold

        for i in range(min_keep, min(len(sorted_sims) - 1, 200)):
            gap = sorted_sims[i] - sorted_sims[i + 1]
            if gap > best_gap and sorted_sims[i + 1] >= min_threshold:
                best_gap = gap
                # Set threshold just below the gap
                best_threshold = sorted_sims[i + 1] + 0.01

        # If no significant gap found, use a percentile-based approach
        if best_gap < 0.05:
            # Use the 75th percentile of scores as threshold
            idx = min(len(sorted_sims) - 1, int(len(sorted_sims) * 0.75))
            best_threshold = max(min_threshold, sorted_sims[idx])

        # Clamp to valid range
        return max(min_threshold, min(max_threshold, best_threshold))

    def _filter_by_relevance(
        self,
        query_word: str,
        candidates: list[dict[str, Any]],
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Filter candidates by relevance to query word.

        Args:
            query_word: The query word
            candidates: List of candidate word data
            threshold: Similarity threshold (None = adaptive)

        Returns:
            Filtered list of candidates with sufficient relevance
        """
        # Compute relevance scores
        scored = self._compute_query_relevance(query_word, candidates)

        if not scored:
            return []

        # Extract similarities for threshold calculation
        similarities = [sim for _, sim in scored]

        # Determine threshold
        if threshold is None:
            # Adaptive threshold
            threshold = self._find_adaptive_threshold(similarities)
            print(f"Adaptive relevance threshold: {threshold:.3f}")
        else:
            print(f"Using relevance threshold: {threshold:.3f}")

        # Filter by threshold with tiered approach:
        # - Datamuse synonyms (rel_syn): always kept (high-quality curated)
        # - Moby / ConceptNet: lower floor (0.25) but still filtered
        # - Uncurated ML results need higher threshold to filter tangential words
        ml_threshold = max(threshold + 0.05, 0.30)  # Stricter for ML-only results
        curated_threshold = max(threshold - 0.05, 0.15)  # Floor for Moby / ConceptNet single words

        # Phrases need higher similarity threshold since average embeddings are noisier
        phrase_threshold = max(threshold, 0.55)
        # Curated phrases get a lower but non-zero threshold
        curated_phrase_threshold = 0.30

        filtered = []
        for candidate, sim in scored:
            tags = candidate.get("tags", [])
            is_synonym = "syn" in tags  # Datamuse explicit synonym
            is_moby = "moby" in tags    # From Moby Thesaurus
            is_conceptnet = "conceptnet" in tags  # From ConceptNet
            word = candidate.get("word", "")
            is_phrase = " " in word

            has_embedding = candidate.get("_has_embedding", True)

            # Words not in GloVe vocabulary (archaic/rare) need special handling
            # They get default scores (0.5 for synonyms, 0.25 otherwise) which aren't reliable
            if not has_embedding:
                # Datamuse synonyms without embeddings: keep only if explicitly marked
                # (these are modern-enough words that Datamuse knows about)
                if is_synonym:
                    candidate["_relevance"] = sim
                    filtered.append(candidate)
                    continue
                # Moby/ConceptNet words without GloVe embeddings: likely archaic, skip them
                # This filters out "duodecimo" and similar obscure entries
                else:
                    continue

            # Datamuse synonyms with embeddings are always kept (high-quality)
            if is_synonym:
                candidate["_relevance"] = sim
                filtered.append(candidate)
                continue

            # Determine the effective threshold for this candidate
            if is_phrase:
                effective_threshold = curated_phrase_threshold if (is_moby or is_conceptnet) else phrase_threshold
            elif is_moby or is_conceptnet:
                effective_threshold = curated_threshold
            else:
                effective_threshold = ml_threshold

            if sim >= effective_threshold:
                # Store the relevance score for later use
                candidate["_relevance"] = sim
                filtered.append(candidate)

        print(f"Relevance filter: {len(candidates)} → {len(filtered)} candidates (threshold={threshold:.3f})")
        return filtered

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
            freq = None
            for tag in tags:
                if tag.startswith("f:"):
                    try:
                        freq = float(tag[2:])
                    except ValueError:
                        pass
                    break

            # Moby/ConceptNet entries are always treated as common
            # (they're from curated sources, so they're legitimate words)
            is_moby = "moby" in tags
            is_conceptnet = "conceptnet" in tags
            is_curated_source = is_moby or is_conceptnet
            if freq is None:
                freq = 0.5 if is_curated_source else 0.0

            # Curated words always go to common, regardless of frequency
            if is_curated_source or freq >= common_threshold:
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
        max_clusters: int = 10,
        query_word: str = "",
    ) -> tuple[list[dict[str, Any]], list[int] | None, list[str] | None]:
        """Sample results to ensure diversity across semantic clusters.

        Uses blended GloVe + Moby co-synonym similarity for clustering.

        Args:
            results: All candidate results
            limit: Target number of results to return
            min_per_cluster: Minimum words per cluster (if available)
            max_clusters: Maximum number of clusters to create
            query_word: Original query word

        Returns:
            Tuple of (sampled_results, cluster_assignments, cluster_labels)
        """
        if not SKLEARN_AVAILABLE or self._embeddings is None or self._word2idx is None:
            if len(results) <= limit:
                return results, None, None
            return results[:limit], None, None

        needs_sampling = len(results) > limit
        word_texts = [w.get("word", "").lower() for w in results]

        # Ensure single-word diversity: reserve slots for single words
        # This prevents phrases from dominating the results
        single_word_indices = [i for i, w in enumerate(word_texts) if " " not in w]
        phrase_indices = [i for i, w in enumerate(word_texts) if " " in w]

        min_single_words = min(limit // 2, len(single_word_indices))  # At least half should be single words
        if needs_sampling and len(single_word_indices) > min_single_words:
            # Prioritize single words, then fill with phrases
            priority_indices = single_word_indices[:min_single_words]
            remaining_singles = single_word_indices[min_single_words:]
            remaining_slots = limit - min_single_words
            # Mix remaining singles and phrases
            other_indices = remaining_singles + phrase_indices
            priority_indices.extend(other_indices[:remaining_slots])
            # Rebuild results in this order
            priority_indices = sorted(set(priority_indices))
            results = [results[i] for i in priority_indices]
            word_texts = [w.get("word", "").lower() for w in results]
            needs_sampling = len(results) > limit

        if len(word_texts) < 5:
            if len(results) <= limit:
                return results, None, None
            return results[:limit], None, None

        dist = self._build_distance_matrix(word_texts)
        actual_n_clusters = self._find_optimal_clusters(dist, min_k=2, max_k=max_clusters)

        ac = AgglomerativeClustering(
            n_clusters=actual_n_clusters, metric="precomputed", linkage="average"
        )
        cluster_assignments = [int(x) for x in ac.fit_predict(dist)]

        # Merge small clusters — scale min_size to dataset so small result sets
        # don't collapse into a single cluster
        adaptive_min_size = max(3, len(word_texts) // (actual_n_clusters * 2))
        cluster_assignments = self._merge_small_clusters(cluster_assignments, dist, min_size=adaptive_min_size)

        # Build cluster -> indices mapping
        clusters: dict[int, list[int]] = {}
        for i, c in enumerate(cluster_assignments):
            if c not in clusters:
                clusters[c] = []
            clusters[c].append(i)

        # Sample from each cluster (only if we have more results than limit)
        if needs_sampling:
            sampled_indices: list[int] = []
            remaining = limit

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

            # Second pass: fill remaining proportionally
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

            sampled_indices = sorted(set(sampled_indices))[:limit]
            sampled_results = [results[i] for i in sampled_indices]
            sampled_assignments = [cluster_assignments[i] for i in sampled_indices]
        else:
            sampled_results = results
            sampled_assignments = cluster_assignments

        # Remap cluster IDs to be contiguous (0, 1, 2, ...)
        unique_clusters = sorted(set(c for c in sampled_assignments if c >= 0))
        cluster_remap = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
        sampled_assignments = [cluster_remap.get(c, 0) for c in sampled_assignments]

        # Generate descriptive cluster labels
        cluster_label_names = self._generate_cluster_labels(
            sampled_results, sampled_assignments, len(unique_clusters)
        )

        print(f"Clustering: {len(unique_clusters)} clusters from {len(word_texts)} words (hybrid GloVe+Moby)")

        return sampled_results, sampled_assignments, cluster_label_names

    def search(
        self,
        word: str,
        sense: str | None = None,
        limit: int = 100,
        layout: str = "sectors",
        include_rare: bool = False,
        relevance: float | None = None,
    ) -> SearchResult | None:
        """Search for semantically related words.

        Args:
            word: The word to search for
            sense: Optional specific sense (not used currently)
            limit: Maximum results to return
            layout: Visualization layout
            include_rare: Whether to include rare words
            relevance: Similarity threshold (0.0-1.0). None = adaptive.
                       Higher values = stricter filtering (only close synonyms).
                       Lower values = looser filtering (more related words).

        Returns:
            SearchResult with neighbors and clusters
        """
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

        # Add results from Moby Thesaurus (good for literary/poetic phrases)
        # Also tag existing Datamuse words with "moby" if they appear in Moby
        moby = get_moby_thesaurus()
        moby_synonyms = moby.get_synonyms(normalized)
        moby_set = set(s.lower() for s in moby_synonyms)

        # Tag existing results that are also in Moby
        moby_tagged = 0
        for result in results:
            word_lower = result.get("word", "").lower()
            if word_lower in moby_set:
                if "tags" not in result:
                    result["tags"] = []
                if "moby" not in result["tags"]:
                    result["tags"].append("moby")
                    moby_tagged += 1

        # Add new Moby words not already in results
        moby_added = 0
        for moby_word in moby_synonyms:
            if moby_word.lower() not in seen_words:
                results.append({
                    "word": moby_word,
                    "score": 500,  # Moderate score
                    "tags": ["moby"],
                })
                seen_words.add(moby_word.lower())
                moby_added += 1

        if moby_added > 0 or moby_tagged > 0:
            print(f"Moby Thesaurus: {moby_added} new words, {moby_tagged} existing words tagged")

        # Add results from ConceptNet (Synonym + RelatedTo edges)
        conceptnet_candidates = self._fetch_from_conceptnet(normalized)
        conceptnet_added = 0
        conceptnet_tagged = 0
        for cn_word in conceptnet_candidates:
            word_text = cn_word.get("word", "").lower()
            if word_text in seen_words:
                # Tag existing result with "conceptnet"
                for result in results:
                    if result.get("word", "").lower() == word_text:
                        if "tags" not in result:
                            result["tags"] = []
                        if "conceptnet" not in result["tags"]:
                            result["tags"].append("conceptnet")
                            conceptnet_tagged += 1
                        break
            else:
                results.append(cn_word)
                seen_words.add(word_text)
                conceptnet_added += 1

        if conceptnet_added > 0 or conceptnet_tagged > 0:
            print(f"ConceptNet: {conceptnet_added} new words, {conceptnet_tagged} existing words tagged")

        if not results:
            return None

        # Note: Multi-word phrases are now handled in _filter_by_relevance
        # using average embeddings of component words

        # Filter by relevance to query word (removes loosely related words)
        # This is the key step that filters out "asia", "japan" for "ocean"
        results = self._filter_by_relevance(normalized, results, threshold=relevance)

        if not results:
            return None

        # Filter by frequency (get more than limit to allow smart sampling)
        frequency_limit = limit * 3 if self._embeddings is not None else limit
        results = self._filter_by_frequency(
            results, include_rare=include_rare, limit=frequency_limit, rare_bonus=150
        )

        # Smart sample to ensure diversity across semantic clusters
        # Returns precomputed cluster info to avoid duplicate clustering
        results, precomputed_assignments, precomputed_labels = self._smart_sample(
            results, limit=limit, max_clusters=12, query_word=normalized
        )

        if not results:
            return None

        # Compute visualization data (reuse cluster info from sampling if available)
        cluster_assignments, custom_cluster_labels = self._assign_clusters(
            results, precomputed_assignments, precomputed_labels, query_word=normalized
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
                formality=compute_formality(normalized),
            )
        )

        for i, result in enumerate(results):
            result_word = result.get("word", "")

            # Use GloVe relevance for similarity (determines font weight in UI)
            # This is more meaningful than Datamuse score (which is 0 for synonyms)
            relevance = result.get("_relevance", 0.5)

            # Cap at 1.0 and ensure reasonable display range
            similarity = min(1.0, relevance)

            neighbors.append(
                WordResult(
                    word=result_word,
                    similarity=round(similarity, 3),
                    coordinates=coordinates[i] if i < len(coordinates) else (0.5, 0.5),
                    frequency=self._get_frequency_tier(result),
                    cluster=cluster_assignments[i] if i < len(cluster_assignments) else 0,
                    formality=compute_formality(result_word),
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
                if w not in seen_words and w not in exclude_set:
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
                    vec = None

                    if w in self._word2idx:
                        # Single word with embedding
                        idx = self._word2idx[w]
                        vec = self._embeddings[idx]
                    elif " " in w:
                        # Multi-word phrase - compute average embedding
                        component_vecs = []
                        for component in w.split():
                            if component in self._word2idx:
                                idx = self._word2idx[component]
                                component_vecs.append(self._embeddings[idx])
                        if component_vecs:
                            vec = np.mean(component_vecs, axis=0)

                    if vec is not None:
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
        cluster_assignments, custom_labels = self._assign_clusters(results, query_word=normalized)
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
                    frequency=self._get_frequency_tier(result),
                    cluster=cluster_assignments[i] if i < len(cluster_assignments) else 0,
                    formality=compute_formality(result_word),
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
