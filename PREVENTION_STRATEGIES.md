# Prevention Strategies for Semantic Word Explorer

This document outlines best practices, testing strategies, and design principles to prevent common issues when working with multi-API data integration, filter toggles, and visualization systems.

---

## Table of Contents

1. [API Score Normalization](#1-api-score-normalization)
2. [Additive Filter Design](#2-additive-filter-design)
3. [Distant Relationship Pollution](#3-distant-relationship-pollution)
4. [Visualization Collision](#4-visualization-collision)
5. [Implementation Checklist](#5-implementation-checklist)
6. [Testing Strategies](#6-testing-strategies)

---

## 1. API Score Normalization

### The Problem

When combining results from multiple APIs (e.g., Datamuse synonyms + "means like" results), different endpoints return scores on different scales. Directly comparing or merging these scores creates misleading rankings.

**Current Implementation Reference:**
- `/Users/bmo/Documents/coding projects/semantic-word-explorer/backend/app/embeddings/datamuse_provider.py` (lines 498-629)

### Best Practices

#### 1.1 Use Rank-Based Positioning

When exact score comparison is not meaningful, use position/rank instead:

```python
# GOOD: Rank-based approach
def merge_results_by_rank(primary: list, secondary: list) -> list:
    """Merge by interleaving based on source rank, not absolute score."""
    merged = []
    seen = set()

    # Primary results get priority ranks 0, 2, 4, ...
    # Secondary results get ranks 1, 3, 5, ...
    p_idx, s_idx = 0, 0

    while p_idx < len(primary) or s_idx < len(secondary):
        # Alternate between sources
        if p_idx < len(primary):
            item = primary[p_idx]
            if item['word'] not in seen:
                merged.append(item)
                seen.add(item['word'])
            p_idx += 1

        if s_idx < len(secondary):
            item = secondary[s_idx]
            if item['word'] not in seen:
                merged.append(item)
                seen.add(item['word'])
            s_idx += 1

    return merged
```

#### 1.2 Normalize Within Each Source First

Before combining, normalize scores to a 0-1 range within each API's results:

```python
# GOOD: Source-local normalization
def normalize_scores(results: list[dict]) -> list[dict]:
    """Normalize scores to 0-1 within a single source."""
    if not results:
        return results

    scores = [r.get('score', 0) for r in results]
    max_score = max(scores) if scores else 1
    min_score = min(scores) if scores else 0
    score_range = max_score - min_score

    if score_range == 0:
        # All same score - assign based on position
        return [
            {**r, 'normalized_score': 1 - (i / len(results))}
            for i, r in enumerate(results)
        ]

    return [
        {**r, 'normalized_score': (r.get('score', 0) - min_score) / score_range}
        for r in results
    ]

# BAD: Direct cross-source score comparison
def bad_merge(source_a: list, source_b: list) -> list:
    combined = source_a + source_b
    return sorted(combined, key=lambda x: x['score'], reverse=True)  # WRONG!
```

#### 1.3 Apply Source Weighting

When sources have different reliability, apply explicit weights:

```python
# GOOD: Weighted source combination
SOURCE_WEIGHTS = {
    'synonyms': 1.0,      # Explicit synonyms are most reliable
    'means_like': 0.7,    # Semantic similarity is good but broader
    'sounds_like': 0.3,   # Phonetic similarity is less relevant
}

def weighted_merge(results_by_source: dict[str, list]) -> list:
    """Combine results with source-specific weighting."""
    all_results = []

    for source, results in results_by_source.items():
        weight = SOURCE_WEIGHTS.get(source, 0.5)
        normalized = normalize_scores(results)

        for r in normalized:
            all_results.append({
                **r,
                'final_score': r['normalized_score'] * weight,
                'source': source
            })

    # Deduplicate, keeping highest final_score
    seen = {}
    for r in all_results:
        word = r['word']
        if word not in seen or r['final_score'] > seen[word]['final_score']:
            seen[word] = r

    return sorted(seen.values(), key=lambda x: x['final_score'], reverse=True)
```

### Testing for Score Normalization

```python
def test_score_normalization_consistency():
    """Verify normalized scores are always in 0-1 range."""
    results = provider.search("happy")

    for neighbor in results.neighbors:
        assert 0 <= neighbor.similarity <= 1, \
            f"Score out of range: {neighbor.word} = {neighbor.similarity}"

def test_multi_source_ordering():
    """Verify synonyms appear before less relevant results."""
    results = provider.search("happy")

    # Find first synonym and first non-synonym
    first_synonym_idx = None
    first_other_idx = None

    for i, n in enumerate(results.neighbors):
        if 'syn' in n.tags and first_synonym_idx is None:
            first_synonym_idx = i
        elif 'syn' not in n.tags and first_other_idx is None:
            first_other_idx = i

    if first_synonym_idx is not None and first_other_idx is not None:
        assert first_synonym_idx < first_other_idx, \
            "Synonyms should appear before related words"
```

---

## 2. Additive Filter Design

### The Problem

Toggle filters labeled "include more" should expand results, not replace them. Users expect `includeRare=true` to show everything from `includeRare=false` **plus** additional rare words.

**Current Implementation Reference:**
- `/Users/bmo/Documents/coding projects/semantic-word-explorer/backend/app/embeddings/datamuse_provider.py` (lines 446-496)

### Best Practices

#### 2.1 Superset Guarantee Pattern

Structure filter logic to guarantee the expanded mode is a superset:

```python
# GOOD: Additive filter design
def filter_by_frequency(
    results: list,
    include_rare: bool = False,
    common_limit: int = 100,
    rare_bonus: int = 50
) -> list:
    """
    Additive filter: include_rare=True returns everything from
    include_rare=False PLUS additional rare words.
    """
    # Separate into categories
    common = [r for r in results if is_common(r)]
    rare = [r for r in results if not is_common(r)]

    # Base set: common words up to limit
    base_results = common[:common_limit]

    if include_rare:
        # ADDITIVE: Base + rare bonus
        return base_results + rare[:rare_bonus]
    else:
        return base_results

# BAD: Replacement filter (breaks user expectations)
def bad_filter(results: list, include_rare: bool = False) -> list:
    if include_rare:
        return results  # Returns everything
    else:
        return [r for r in results if is_common(r)]  # Different set!
    # Problem: Rare mode might exclude some common words that were included
    # if the total limit is reached before all common words are added
```

#### 2.2 Document Filter Semantics Clearly

```python
class FilterConfig:
    """Filter configuration with clear additive semantics.

    Design Principle: Every filter toggle labeled "include X" or
    "show X" MUST be additive. The enabled state always includes
    everything from the disabled state.

    Invariant: filter(params, feature=True) >= filter(params, feature=False)
    """

    include_rare: bool = False      # Adds rare words to common words
    include_archaic: bool = False   # Adds archaic words to modern words
    include_technical: bool = False # Adds technical terms to general terms
```

#### 2.3 Implement Filter Composition

```python
# GOOD: Composable additive filters
class AdditiveFilter:
    """Filter that guarantees superset relationship."""

    def __init__(self, base_predicate, name: str):
        self.base_predicate = base_predicate
        self.name = name

    def apply(self, items: list, include_extra: bool) -> list:
        """Apply filter additively.

        Returns:
            include_extra=False: Items matching base_predicate
            include_extra=True:  All items (superset guaranteed)
        """
        if include_extra:
            # Return all items, but sort so base matches come first
            base = [i for i in items if self.base_predicate(i)]
            extra = [i for i in items if not self.base_predicate(i)]
            return base + extra
        else:
            return [i for i in items if self.base_predicate(i)]

# Usage
frequency_filter = AdditiveFilter(
    base_predicate=lambda w: w.frequency_score > 0.5,
    name="common_words"
)

# This guarantees:
# frequency_filter.apply(words, include_extra=True)
#   is a superset of
# frequency_filter.apply(words, include_extra=False)
```

### Testing for Additive Filters

```python
def test_include_rare_is_superset():
    """Verify includeRare=True returns superset of includeRare=False."""
    results_common = provider.search("happy", include_rare=False)
    results_with_rare = provider.search("happy", include_rare=True)

    common_words = {n.word for n in results_common.neighbors}
    all_words = {n.word for n in results_with_rare.neighbors}

    # Every word in common mode must be in rare mode
    missing = common_words - all_words
    assert not missing, \
        f"Words in common mode missing from rare mode: {missing}"

    # Rare mode should have at least as many words
    assert len(all_words) >= len(common_words), \
        "Rare mode should have equal or more results"

def test_filter_additivity_property():
    """Property-based test for filter additivity."""
    for word in ["happy", "bank", "run", "light"]:
        restricted = provider.search(word, include_rare=False)
        expanded = provider.search(word, include_rare=True)

        restricted_set = {n.word for n in restricted.neighbors}
        expanded_set = {n.word for n in expanded.neighbors}

        assert restricted_set.issubset(expanded_set), \
            f"Additivity violated for '{word}': " \
            f"restricted has {restricted_set - expanded_set} not in expanded"
```

---

## 3. Distant Relationship Pollution

### The Problem

Fetching too many results from semantic APIs pulls in distantly related terms that add noise. "Happy" shouldn't show "weather" just because both can describe a "sunny day."

**Current Implementation Reference:**
- `/Users/bmo/Documents/coding projects/semantic-word-explorer/backend/app/embeddings/datamuse_provider.py` (lines 528-533)

### Best Practices

#### 3.1 Limit API Fetch Sizes

```python
# GOOD: Conservative fetch limits
FETCH_LIMITS = {
    'synonyms': 100,      # Explicit synonyms are high quality
    'means_like': 150,    # Broader but still relevant
    'sounds_like': 50,    # Only useful for specific features
    'triggers': 30,       # Association is very broad
}

def search_with_limits(word: str) -> list:
    """Fetch with appropriate limits per relationship type."""
    synonyms = fetch_api(rel='syn', word=word, max=FETCH_LIMITS['synonyms'])
    means_like = fetch_api(rel='ml', word=word, max=FETCH_LIMITS['means_like'])

    # Don't fetch very broad relationships at all for core semantic search
    # triggers = fetch_api(rel='trg', word=word, max=FETCH_LIMITS['triggers'])

    return merge_by_rank(synonyms, means_like)
```

#### 3.2 Apply Relevance Thresholds

```python
# GOOD: Score thresholds to filter weak relationships
def filter_by_relevance(results: list, min_score_ratio: float = 0.1) -> list:
    """Remove results with scores far below the maximum."""
    if not results:
        return results

    max_score = max(r.get('score', 0) for r in results)
    threshold = max_score * min_score_ratio

    return [r for r in results if r.get('score', 0) >= threshold]

# Example: If top result has score 1000, filter out anything below 100
```

#### 3.3 Prioritize Direct Relationships

```python
# GOOD: Source priority with cutoffs
def merge_with_priority(
    synonyms: list,
    related: list,
    max_total: int = 100
) -> list:
    """Merge prioritizing direct relationships with hard cutoffs."""

    # Always include all synonyms (up to reasonable limit)
    result = synonyms[:50]
    seen = {r['word'] for r in result}

    # Fill remaining slots with related words
    remaining = max_total - len(result)

    for r in related:
        if remaining <= 0:
            break
        if r['word'] not in seen:
            result.append(r)
            seen.add(r['word'])
            remaining -= 1

    return result
```

#### 3.4 Use Relationship-Specific Multipliers

```python
# GOOD: Limit based on relationship type strength
def calculate_fetch_limit(
    base_limit: int,
    relationship_type: str
) -> int:
    """Calculate appropriate fetch limit based on relationship strength."""

    MULTIPLIERS = {
        'synonym': 1.0,        # Direct synonyms - fetch full amount
        'means_like': 1.5,     # Semantic similarity - fetch more, filter later
        'similar_to': 0.8,     # Often redundant with means_like
        'triggers': 0.3,       # Very broad associations
        'antonym': 0.5,        # Useful but limited quantity
    }

    multiplier = MULTIPLIERS.get(relationship_type, 0.5)
    return int(base_limit * multiplier)
```

### Testing for Distant Relationships

```python
def test_no_distant_pollution():
    """Verify results are meaningfully related to query word."""
    results = provider.search("happy")

    # Check that all results have reasonable similarity scores
    for neighbor in results.neighbors:
        if neighbor.word != "happy":
            assert neighbor.similarity >= 0.1, \
                f"Distantly related word included: {neighbor.word} " \
                f"(similarity={neighbor.similarity})"

def test_fetch_limit_respected():
    """Verify API fetches don't exceed configured limits."""
    with patch('httpx.Client.get') as mock_get:
        mock_get.return_value.json.return_value = []

        provider.search("test")

        # Check all API calls used appropriate limits
        for call in mock_get.call_args_list:
            params = call.kwargs.get('params', {})
            max_param = int(params.get('max', 1000))
            assert max_param <= 200, \
                f"API fetch limit too high: {max_param}"
```

---

## 4. Visualization Collision

### The Problem

Text labels overlap when points are placed too close together, making the visualization unreadable.

**Current Implementation Reference:**
- `/Users/bmo/Documents/coding projects/semantic-word-explorer/backend/app/embeddings/datamuse_provider.py` (lines 65-123)
- `/Users/bmo/Documents/coding projects/semantic-word-explorer/src/components/WordScatterPlot.tsx`

### Best Practices

#### 4.1 Always Apply Collision Avoidance Post-Processing

```python
# GOOD: Mandatory collision avoidance step
def compute_layout(words: list, layout: str) -> list[tuple[float, float]]:
    """Compute coordinates with guaranteed collision avoidance."""

    # Step 1: Compute initial layout
    if layout == "sectors":
        coords = layout_sectors(words)
    elif layout == "rings":
        coords = layout_rings(words)
    else:
        coords = layout_default(words)

    # Step 2: ALWAYS apply collision avoidance (not optional!)
    coords = avoid_collisions(coords, min_distance=0.04, iterations=20)

    return coords
```

#### 4.2 Implement Iterative Repulsion

```python
def avoid_collisions(
    coordinates: list[tuple[float, float]],
    min_distance: float = 0.04,
    iterations: int = 20
) -> list[tuple[float, float]]:
    """Push overlapping points apart iteratively."""

    if len(coordinates) < 2:
        return coordinates

    coords = [list(c) for c in coordinates]
    n = len(coords)

    for _ in range(iterations):
        # Push points away from center (where query word sits)
        for i in range(n):
            dx = coords[i][0] - 0.5
            dy = coords[i][1] - 0.5
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < min_distance and dist > 0.001:
                push = (min_distance - dist) * 0.5
                coords[i][0] += (dx / dist) * push
                coords[i][1] += (dy / dist) * push

        # Push points apart from each other
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

    # Clamp to valid range
    return [
        (max(0.02, min(0.98, x)), max(0.02, min(0.98, y)))
        for x, y in coords
    ]
```

#### 4.3 Client-Side Label Collision Handling

For additional protection, handle collisions in the visualization layer:

```typescript
// GOOD: Client-side collision detection
interface LabelBounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

function adjustLabelPositions(
  labels: LabelBounds[],
  minGap: number = 5
): LabelBounds[] {
  const adjusted = labels.map(l => ({ ...l }));

  for (let iteration = 0; iteration < 10; iteration++) {
    for (let i = 0; i < adjusted.length; i++) {
      for (let j = i + 1; j < adjusted.length; j++) {
        const overlap = getOverlap(adjusted[i], adjusted[j]);

        if (overlap > 0) {
          // Push labels apart
          const dx = adjusted[j].x - adjusted[i].x;
          const dy = adjusted[j].y - adjusted[i].y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;

          const push = (overlap + minGap) / 2;
          adjusted[i].x -= (dx / dist) * push;
          adjusted[i].y -= (dy / dist) * push;
          adjusted[j].x += (dx / dist) * push;
          adjusted[j].y += (dy / dist) * push;
        }
      }
    }
  }

  return adjusted;
}
```

#### 4.4 Use Text Anchoring Strategies

```typescript
// GOOD: Dynamic text anchoring based on position
function getTextAnchor(x: number, y: number): {
  anchor: 'start' | 'middle' | 'end';
  baseline: 'top' | 'middle' | 'bottom';
} {
  // Anchor text away from edges and center
  const anchor = x < 0.3 ? 'start' : x > 0.7 ? 'end' : 'middle';
  const baseline = y < 0.3 ? 'top' : y > 0.7 ? 'bottom' : 'middle';

  return { anchor, baseline };
}
```

### Testing for Visualization Collisions

```python
def test_no_coordinate_collisions():
    """Verify no two words have overlapping coordinates."""
    results = provider.search("happy", limit=100)

    MIN_DISTANCE = 0.03
    coordinates = [(n.coordinates.x, n.coordinates.y) for n in results.neighbors]

    for i, (x1, y1) in enumerate(coordinates):
        for j, (x2, y2) in enumerate(coordinates):
            if i >= j:
                continue

            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            assert distance >= MIN_DISTANCE, \
                f"Collision between words at indices {i} and {j}: " \
                f"distance={distance:.4f} < {MIN_DISTANCE}"

def test_center_avoidance():
    """Verify words don't overlap with center query word."""
    results = provider.search("happy")

    CENTER = (0.5, 0.5)
    MIN_CENTER_DISTANCE = 0.04

    for neighbor in results.neighbors:
        if neighbor.word == "happy":
            continue  # Skip query word itself

        distance = math.sqrt(
            (neighbor.coordinates.x - CENTER[0])**2 +
            (neighbor.coordinates.y - CENTER[1])**2
        )
        assert distance >= MIN_CENTER_DISTANCE, \
            f"Word '{neighbor.word}' too close to center: {distance:.4f}"
```

---

## 5. Implementation Checklist

Use this checklist when implementing new features:

### API Integration
- [ ] Scores are normalized within each source before combining
- [ ] Rank-based ordering is used when score comparison is meaningless
- [ ] Source weights are documented and configurable
- [ ] API fetch limits are configured per relationship type

### Filter Toggles
- [ ] "Include X" filters are documented as additive
- [ ] Superset property is explicitly tested
- [ ] Filter logic separates base set from bonus set
- [ ] User-facing labels match filter behavior ("Include rare" not "Show only rare")

### Data Quality
- [ ] Fetch limits prevent distant relationship pollution
- [ ] Relevance thresholds filter weak associations
- [ ] Direct relationships (synonyms) are prioritized
- [ ] Multi-word phrases are filtered for visualization

### Visualization
- [ ] Collision avoidance is applied after all layouts
- [ ] Minimum distance constants are defined
- [ ] Center (query word) has protected space
- [ ] Coordinates are clamped to valid display range

---

## 6. Testing Strategies

### Unit Tests

```python
# tests/test_normalization.py
class TestScoreNormalization:
    """Tests for API score normalization."""

    def test_normalized_scores_in_range(self):
        """All normalized scores should be 0-1."""
        pass

    def test_max_score_normalized_to_one(self):
        """Highest score should normalize to 1.0."""
        pass

    def test_relative_ordering_preserved(self):
        """Normalization should preserve relative ordering."""
        pass


# tests/test_filters.py
class TestAdditiveFilters:
    """Tests for filter additivity properties."""

    def test_include_rare_superset(self):
        """include_rare=True is superset of include_rare=False."""
        pass

    def test_filter_composition(self):
        """Multiple filters compose additively."""
        pass


# tests/test_visualization.py
class TestVisualization:
    """Tests for visualization coordinate generation."""

    def test_no_collisions(self):
        """No coordinates are within min_distance of each other."""
        pass

    def test_center_protected(self):
        """Center area reserved for query word."""
        pass
```

### Property-Based Tests

```python
from hypothesis import given, strategies as st

@given(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=100))
def test_collision_avoidance_always_separates(word_list):
    """Property: collision avoidance always produces valid separations."""
    initial_coords = [(random.random(), random.random()) for _ in word_list]
    final_coords = avoid_collisions(initial_coords, min_distance=0.04)

    for i, (x1, y1) in enumerate(final_coords):
        for j, (x2, y2) in enumerate(final_coords):
            if i >= j:
                continue
            distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            # Allow small tolerance for edge cases
            assert distance >= 0.03, f"Collision at indices {i}, {j}"
```

### Integration Tests

```python
class TestAPIIntegration:
    """Integration tests for full API flow."""

    def test_end_to_end_search(self):
        """Full search returns valid, well-structured results."""
        response = client.get("/explore?word=happy")
        assert response.status_code == 200

        data = response.json()
        assert "neighbors" in data
        assert "clusters" in data

        # Verify all scores normalized
        for neighbor in data["neighbors"]:
            assert 0 <= neighbor["similarity"] <= 1

        # Verify no collisions
        coords = [(n["coordinates"]["x"], n["coordinates"]["y"])
                  for n in data["neighbors"]]
        # ... collision check

    def test_filter_additivity_integration(self):
        """Verify filter additivity in full API responses."""
        common = client.get("/explore?word=happy&include_rare=false").json()
        rare = client.get("/explore?word=happy&include_rare=true").json()

        common_words = {n["word"] for n in common["neighbors"]}
        rare_words = {n["word"] for n in rare["neighbors"]}

        assert common_words.issubset(rare_words)
```

### Visual Regression Tests

For visualization changes, use screenshot comparison:

```typescript
// cypress/e2e/visualization.cy.ts
describe('Word Visualization', () => {
  it('should render without label overlaps', () => {
    cy.visit('/explore/happy');
    cy.wait('@exploreAPI');

    // Visual regression snapshot
    cy.get('.scatter-plot').matchImageSnapshot('happy-visualization');
  });

  it('should show more words when rare filter enabled', () => {
    cy.visit('/explore/happy');
    cy.get('[data-testid="include-rare"]').click();

    // Count visible labels
    cy.get('.word-label').should('have.length.greaterThan', 50);
  });
});
```

---

## Summary

| Issue | Prevention Strategy | Key Test |
|-------|-------------------|----------|
| Score normalization | Normalize within source, use rank-based merging | `assert 0 <= score <= 1` |
| Additive filters | Base set + bonus pattern | `assert restricted.issubset(expanded)` |
| Distant pollution | Conservative fetch limits, relevance thresholds | `assert similarity >= 0.1` |
| Visualization collision | Mandatory post-processing repulsion | `assert distance >= MIN_DISTANCE` |
