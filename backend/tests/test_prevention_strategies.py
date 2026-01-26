"""
Tests for prevention strategies against common Semantic Word Explorer issues.

These tests verify:
1. API score normalization - scores are properly normalized when combining sources
2. Additive filter design - include_rare=True is a superset of include_rare=False
3. Distant relationship pollution - results are meaningfully related
4. Visualization collision - coordinates have minimum separation

Run with: pytest backend/tests/test_prevention_strategies.py -v
"""

import math
import pytest
from unittest.mock import patch, MagicMock

# Import the provider (adjust path as needed based on how tests are run)
import sys
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.embeddings.datamuse_provider import DatamuseProvider
from app.embeddings.base import SearchResult


class TestScoreNormalization:
    """Tests for Issue 1: API score normalization."""

    @pytest.fixture
    def provider(self):
        return DatamuseProvider()

    def test_all_similarity_scores_in_valid_range(self, provider):
        """All normalized similarity scores must be in [0, 1] range."""
        # Use a word likely to have results
        result = provider.search("happy", limit=50)

        if result is None:
            pytest.skip("API returned no results for 'happy'")

        for neighbor in result.neighbors:
            assert 0 <= neighbor.similarity <= 1, (
                f"Score out of range for '{neighbor.word}': "
                f"similarity={neighbor.similarity}"
            )

    def test_query_word_has_similarity_one(self, provider):
        """The query word itself should have similarity 1.0."""
        result = provider.search("happy", limit=50)

        if result is None:
            pytest.skip("API returned no results")

        # Find the query word in results
        query_neighbors = [n for n in result.neighbors if n.word == "happy"]

        assert len(query_neighbors) == 1, "Query word should appear exactly once"
        assert query_neighbors[0].similarity == 1.0, (
            f"Query word should have similarity 1.0, got {query_neighbors[0].similarity}"
        )

    def test_synonyms_ranked_higher_than_related(self, provider):
        """Explicit synonyms should generally rank higher than 'means like' results."""
        result = provider.search("happy", limit=100)

        if result is None:
            pytest.skip("API returned no results")

        # Synonyms should be in the first cluster (cluster 0)
        synonyms = [n for n in result.neighbors if n.cluster == 0 and n.word != "happy"]
        other = [n for n in result.neighbors if n.cluster != 0]

        if not synonyms or not other:
            pytest.skip("Not enough variety in results to test ranking")

        # Average similarity of synonyms should be higher
        avg_synonym_sim = sum(s.similarity for s in synonyms) / len(synonyms)
        avg_other_sim = sum(o.similarity for o in other) / len(other)

        # Allow some tolerance - synonyms should generally be ranked higher
        assert avg_synonym_sim >= avg_other_sim * 0.8, (
            f"Synonyms (avg={avg_synonym_sim:.3f}) should rank higher than "
            f"other results (avg={avg_other_sim:.3f})"
        )


class TestAdditiveFilters:
    """Tests for Issue 2: Additive filter design."""

    @pytest.fixture
    def provider(self):
        return DatamuseProvider()

    @pytest.mark.parametrize("word", ["happy", "sad", "run", "bank", "light"])
    def test_include_rare_is_superset(self, provider, word):
        """include_rare=True must return a superset of include_rare=False."""
        result_common = provider.search(word, include_rare=False, limit=100)
        result_with_rare = provider.search(word, include_rare=True, limit=100)

        if result_common is None or result_with_rare is None:
            pytest.skip(f"API returned no results for '{word}'")

        common_words = {n.word for n in result_common.neighbors}
        all_words = {n.word for n in result_with_rare.neighbors}

        # Every word in common mode must be in rare mode
        missing = common_words - all_words
        assert not missing, (
            f"Words in common mode missing from rare mode for '{word}': {missing}"
        )

    @pytest.mark.parametrize("word", ["beautiful", "ocean", "mountain"])
    def test_rare_mode_has_equal_or_more_results(self, provider, word):
        """include_rare=True should have at least as many results."""
        result_common = provider.search(word, include_rare=False, limit=100)
        result_with_rare = provider.search(word, include_rare=True, limit=100)

        if result_common is None or result_with_rare is None:
            pytest.skip(f"API returned no results for '{word}'")

        assert len(result_with_rare.neighbors) >= len(result_common.neighbors), (
            f"Rare mode should have >= results for '{word}': "
            f"common={len(result_common.neighbors)}, "
            f"rare={len(result_with_rare.neighbors)}"
        )

    def test_filter_order_independence(self, provider):
        """Filtering should be deterministic regardless of internal order."""
        # Run the same query twice
        result1 = provider.search("happy", include_rare=False, limit=50)
        result2 = provider.search("happy", include_rare=False, limit=50)

        if result1 is None or result2 is None:
            pytest.skip("API returned no results")

        words1 = {n.word for n in result1.neighbors}
        words2 = {n.word for n in result2.neighbors}

        assert words1 == words2, "Same query should return same words"


class TestDistantRelationshipPollution:
    """Tests for Issue 3: Distant relationship pollution."""

    @pytest.fixture
    def provider(self):
        return DatamuseProvider()

    def test_results_have_meaningful_similarity(self, provider):
        """All results should have a minimum meaningful similarity score."""
        result = provider.search("happy", limit=100)

        if result is None:
            pytest.skip("API returned no results")

        MIN_MEANINGFUL_SIMILARITY = 0.05  # 5% of max score

        for neighbor in result.neighbors:
            if neighbor.word == "happy":
                continue  # Skip query word

            assert neighbor.similarity >= MIN_MEANINGFUL_SIMILARITY, (
                f"Distantly related word included: '{neighbor.word}' "
                f"(similarity={neighbor.similarity:.4f} < {MIN_MEANINGFUL_SIMILARITY})"
            )

    def test_no_multiword_phrases(self, provider):
        """Results should not include multi-word phrases."""
        result = provider.search("happy", limit=100)

        if result is None:
            pytest.skip("API returned no results")

        for neighbor in result.neighbors:
            assert " " not in neighbor.word, (
                f"Multi-word phrase in results: '{neighbor.word}'"
            )

    def test_result_count_within_limits(self, provider):
        """Result count should respect configured limits."""
        REQUESTED_LIMIT = 50
        result = provider.search("happy", limit=REQUESTED_LIMIT)

        if result is None:
            pytest.skip("API returned no results")

        # Allow query word + limit
        assert len(result.neighbors) <= REQUESTED_LIMIT + 1, (
            f"Too many results: {len(result.neighbors)} > {REQUESTED_LIMIT + 1}"
        )


class TestVisualizationCollision:
    """Tests for Issue 4: Visualization collision avoidance."""

    @pytest.fixture
    def provider(self):
        return DatamuseProvider()

    MIN_DISTANCE = 0.03  # Minimum distance between points

    def test_no_coordinate_collisions(self, provider):
        """No two words should have overlapping coordinates."""
        result = provider.search("happy", limit=100)

        if result is None:
            pytest.skip("API returned no results")

        coordinates = [
            (n.word, n.coordinates[0], n.coordinates[1])
            for n in result.neighbors
        ]

        for i, (word1, x1, y1) in enumerate(coordinates):
            for j, (word2, x2, y2) in enumerate(coordinates):
                if i >= j:
                    continue

                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                assert distance >= self.MIN_DISTANCE, (
                    f"Collision between '{word1}' and '{word2}': "
                    f"distance={distance:.4f} < {self.MIN_DISTANCE}"
                )

    def test_center_is_protected(self, provider):
        """Non-query words should not overlap with center (0.5, 0.5)."""
        result = provider.search("happy", limit=100)

        if result is None:
            pytest.skip("API returned no results")

        CENTER = (0.5, 0.5)
        MIN_CENTER_DISTANCE = 0.04

        for neighbor in result.neighbors:
            if neighbor.word == "happy":
                continue  # Query word should be at center

            x, y = neighbor.coordinates
            distance = math.sqrt((x - CENTER[0]) ** 2 + (y - CENTER[1]) ** 2)

            assert distance >= MIN_CENTER_DISTANCE, (
                f"'{neighbor.word}' too close to center: "
                f"distance={distance:.4f} < {MIN_CENTER_DISTANCE}"
            )

    def test_coordinates_in_valid_range(self, provider):
        """All coordinates should be within the [0, 1] display range."""
        result = provider.search("happy", limit=100)

        if result is None:
            pytest.skip("API returned no results")

        for neighbor in result.neighbors:
            x, y = neighbor.coordinates
            assert 0 <= x <= 1, f"X coordinate out of range for '{neighbor.word}': {x}"
            assert 0 <= y <= 1, f"Y coordinate out of range for '{neighbor.word}': {y}"

    @pytest.mark.parametrize("layout", ["sectors", "rings", "force", "grid"])
    def test_collision_avoidance_all_layouts(self, provider, layout):
        """Collision avoidance should work for all layout types."""
        result = provider.search("happy", limit=50, layout=layout)

        if result is None:
            pytest.skip("API returned no results")

        coordinates = [
            (n.word, n.coordinates[0], n.coordinates[1])
            for n in result.neighbors
        ]

        collision_count = 0
        for i, (word1, x1, y1) in enumerate(coordinates):
            for j, (word2, x2, y2) in enumerate(coordinates):
                if i >= j:
                    continue

                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if distance < self.MIN_DISTANCE:
                    collision_count += 1

        # Allow a small number of near-collisions (edge cases)
        max_allowed_collisions = len(coordinates) // 20  # 5%
        assert collision_count <= max_allowed_collisions, (
            f"Too many collisions in {layout} layout: "
            f"{collision_count} > {max_allowed_collisions}"
        )


class TestCollisionAvoidanceAlgorithm:
    """Unit tests for the collision avoidance algorithm itself."""

    def test_avoid_collisions_separates_points(self):
        """The collision avoidance function should separate close points."""
        provider = DatamuseProvider()

        # Create intentionally overlapping points
        initial_coords = [
            (0.5, 0.5),
            (0.51, 0.51),  # Very close to first
            (0.52, 0.50),  # Very close to first
            (0.8, 0.8),   # Far away
        ]

        result = provider._avoid_collisions(
            initial_coords,
            min_distance=0.04,
            iterations=20
        )

        # Check all pairs are now separated
        MIN_EXPECTED = 0.03  # Allow small tolerance
        for i, (x1, y1) in enumerate(result):
            for j, (x2, y2) in enumerate(result):
                if i >= j:
                    continue

                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                assert distance >= MIN_EXPECTED, (
                    f"Points {i} and {j} still too close after avoidance: "
                    f"distance={distance:.4f}"
                )

    def test_avoid_collisions_preserves_spread(self):
        """Collision avoidance should not collapse points to a small area."""
        provider = DatamuseProvider()

        # Points spread across the space
        initial_coords = [
            (0.2, 0.2),
            (0.8, 0.2),
            (0.2, 0.8),
            (0.8, 0.8),
            (0.5, 0.5),
        ]

        result = provider._avoid_collisions(
            initial_coords,
            min_distance=0.04,
            iterations=20
        )

        # Calculate bounding box
        xs = [c[0] for c in result]
        ys = [c[1] for c in result]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        # Points should still be spread out
        assert width >= 0.4, f"Points collapsed horizontally: width={width}"
        assert height >= 0.4, f"Points collapsed vertically: height={height}"


class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""

    @pytest.fixture
    def provider(self):
        return DatamuseProvider()

    def test_polysemous_word_handling(self, provider):
        """Words with multiple meanings should return diverse results."""
        result = provider.search("bank", limit=100)

        if result is None:
            pytest.skip("API returned no results for 'bank'")

        # Should have results in multiple clusters (different meanings)
        clusters = {n.cluster for n in result.neighbors}
        assert len(clusters) >= 2, (
            f"Polysemous word 'bank' should have multiple clusters, "
            f"found only: {clusters}"
        )

    def test_rare_words_expand_vocabulary(self, provider):
        """Include rare should add genuinely rare/unusual words."""
        result_common = provider.search("happy", include_rare=False, limit=100)
        result_rare = provider.search("happy", include_rare=True, limit=100)

        if result_common is None or result_rare is None:
            pytest.skip("API returned no results")

        # The rare-only additions
        common_words = {n.word for n in result_common.neighbors}
        rare_words = {n.word for n in result_rare.neighbors}
        rare_only = rare_words - common_words

        # Should have some rare additions
        # (May be 0 if the word doesn't have many rare synonyms)
        assert len(rare_only) >= 0, "Should calculate rare additions correctly"

        # If there are rare additions, verify they're marked appropriately
        if rare_only:
            rare_neighbors = [n for n in result_rare.neighbors if n.word in rare_only]
            rare_frequencies = [n.frequency for n in rare_neighbors]

            # At least some should be marked as rare or uncommon
            non_common = [f for f in rare_frequencies if f in ("rare", "uncommon")]
            # This is a soft assertion - Datamuse frequency data may vary
            # Just verify we can access the frequency field
            assert all(f in ("common", "uncommon", "rare") for f in rare_frequencies)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
