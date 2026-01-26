---
title: "Semantic Word Explorer - API Integration and Result Quality Issues"
date: 2026-01-25
category: integration-issues
tags:
  - datamuse
  - api
  - word-embeddings
  - frequency-filtering
  - visualization
  - glove
  - thesaurus
symptoms:
  - GloVe embeddings returned co-occurrence words instead of true synonyms
  - Medical jargon and rare words overwhelming search results
  - Rare words toggle was exclusive instead of additive
  - Visualization layout broken due to mismatched score scales
  - Distant semantic relationships polluting results (oceanic → kangaroo)
  - Word labels overlapping in visualization
module: semantic-word-explorer
---

# Semantic Word Explorer - API Integration and Result Quality Issues

## Problem Summary

The Semantic Word Explorer thesaurus tool had multiple interrelated issues affecting result quality and visualization:

1. **Wrong similarity type**: GloVe embeddings measure co-occurrence, not semantic similarity
2. **No frequency filtering**: Rare/medical jargon mixed with common words
3. **Broken rare toggle**: Enabling rare words sometimes returned fewer results
4. **Visual clustering**: Words overlapped at center, unreadable
5. **Distant relationships**: "oceanic" returned "kangaroo" through association chains

## Root Cause Analysis

### 1. Mode Confusion
GloVe embeddings produce *contextual* similarity (words in similar contexts) rather than *semantic* similarity (words with similar meanings). "melancholy" returned "instrumentals", "lullaby" instead of "sadness", "sorrow".

### 2. Score Scale Mismatch
Datamuse synonyms API returns scores ~65,000 while the "means like" API returns scores ~39,000,000. Normalizing against the combined max pushed synonyms to the outer edge.

### 3. Non-Additive Filter Logic
The rare toggle used a simple threshold change, meaning rare mode could return a *different* set of words rather than a superset.

### 4. Over-fetching
Requesting `limit * 2` results pulled in loosely related terms via long association chains.

## Solutions

### Fix 1: Dual Provider Architecture

Added Datamuse for semantic similarity alongside GloVe for contextual:

```python
# backend/app/main.py
semantic_provider = get_datamuse_provider()    # True synonyms
contextual_provider = get_embedding_provider()  # Co-occurrence

if mode == "semantic":
    provider = semantic_provider
else:
    provider = contextual_provider
```

### Fix 2: Frequency Filtering with Additive Rare Mode

Separate words into common/rare buckets, with rare mode being *additive*:

```python
# backend/app/embeddings/datamuse_provider.py
def _filter_by_frequency(self, results, include_rare=False,
                         common_threshold=0.5, limit=100, rare_bonus=50):
    common_words = []
    rare_words = []

    for word_data in results:
        freq = extract_frequency(word_data)  # From "f:X.XX" tag
        if freq >= common_threshold:
            common_words.append(word_data)
        else:
            rare_words.append(word_data)

    if include_rare:
        # Additive: all common words + bonus rare words
        return common_words[:limit] + rare_words[:rare_bonus]
    else:
        return common_words[:limit]
```

**Key insight**: `include_rare=True` always returns a superset of `include_rare=False`.

### Fix 3: Rank-Based Positioning

Use position in list (rank) for distance from center, not raw scores:

```python
def _layout_sectors(self, words, clusters):
    for i, word_data in enumerate(words):
        if is_synonym:
            distance = 0.08 + (position_in_cluster / cluster_size) * 0.15
        else:
            distance = 0.25 + (i / n) * 0.20  # Rank-based
```

### Fix 4: Collision Avoidance

Iterative repulsion algorithm pushes overlapping points apart:

```python
def _avoid_collisions(self, coordinates, min_distance=0.04, iterations=20):
    for _ in range(iterations):
        # Push away from center
        for i in range(n):
            if dist_from_center < center_min_dist:
                push_away_from_center(coords[i])

        # Push points apart
        for i, j in combinations:
            if distance(i, j) < min_distance:
                push_apart(coords[i], coords[j])

    return coords
```

### Fix 5: Reduced Fetch Limits

Limited API fetch to prevent distant relationship pollution:

```python
# Was: limit * 2, max 300
# Now: limit * 1.5, max 150
ml_results = self._fetch_from_api({
    "ml": normalized,
    "max": str(min(int(limit * 1.5), 150)),
})
```

## Files Modified

| File | Changes |
|------|---------|
| `backend/app/embeddings/datamuse_provider.py` | New provider with frequency filtering, collision avoidance, rank-based layouts |
| `backend/app/main.py` | Dual provider architecture, mode/layout/includeRare parameters |
| `src/app/explore/[word]/page.tsx` | UI toggles for mode, layout, rare words |
| `src/lib/api.ts` | API client with new parameters |
| `src/components/WordScatterPlot.tsx` | Expanded axis range for collision avoidance |

## Prevention Strategies

1. **API Integration**: When combining results from multiple APIs with different score scales, use rank-based positioning or normalize within each source first.

2. **Additive Toggles**: When implementing "include more" toggles, ensure the expanded mode is always a superset: `expanded_results ⊇ base_results`.

3. **Fetch Limits**: Use conservative fetch limits per relationship type to prevent semantic drift through association chains.

4. **Collision Avoidance**: Always apply collision avoidance post-processing when rendering text labels on scatter plots.

## Testing

```bash
# Test rare mode is additive
curl "http://localhost:8000/explore?word=run&include_rare=false" # → 101 words
curl "http://localhost:8000/explore?word=run&include_rare=true"  # → 119 words (superset)

# Test no distant relationships
curl "http://localhost:8000/explore?word=oceanic" | grep kangaroo  # Should be empty
```

## Results

- **"melancholy"**: Now returns "sadness", "sorrow", "gloom" instead of "instrumentals"
- **"run" without rare**: 101 common words
- **"run" with rare**: 119 words (+18 rare like "scarper", "elope", "rivulet")
- **"oceanic"**: No more "kangaroo" pollution
- **Visualization**: Words spread out with readable labels
