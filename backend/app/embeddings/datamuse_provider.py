"""Datamuse API provider for semantic similarity search.

This provider uses the Datamuse API to find words with similar meanings,
providing true semantic similarity rather than distributional similarity.
"""

import math
import random
from typing import Any

import httpx

from .base import EmbeddingProvider, SearchResult, SenseInfo, WordResult

# Cluster color palette
CLUSTER_COLORS = [
    "#6366f1",  # indigo - synonyms
    "#ec4899",  # pink - related nouns
    "#14b8a6",  # teal - related adjectives
    "#f59e0b",  # amber - related verbs
    "#8b5cf6",  # violet - other
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
    """

    def __init__(self):
        self.api_base = "https://api.datamuse.com/words"
        # Cache for spell checking suggestions
        self._suggestion_cache: dict[str, list[str]] = {}

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

            # Push points apart from each other (gentler)
            for i in range(n):
                for j in range(i + 1, n):
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

    def _assign_clusters(self, words: list[dict[str, Any]]) -> list[int]:
        """Assign cluster IDs based on part of speech.

        Uses the first POS tag in the list as the primary part of speech,
        since Datamuse orders tags by primary usage.
        """
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

        return clusters

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

        # First, fetch explicit synonyms (these are always good)
        synonyms = self._fetch_from_api({
            "rel_syn": normalized,
            "max": str(min(limit, 100)),
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
        # Limit to 1.5x to avoid pulling in distant semantic relationships
        ml_results = self._fetch_from_api({
            "ml": normalized,
            "max": str(min(int(limit * 1.5), 150)),
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

        # Filter by frequency: common words (up to limit), plus rare words if enabled
        results = self._filter_by_frequency(results, include_rare=include_rare, limit=limit)

        if not results:
            return None

        # Compute visualization data
        cluster_assignments = self._assign_clusters(results)
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
                "label": cluster_labels.get(cluster_id, f"cluster {cluster_id}"),
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
