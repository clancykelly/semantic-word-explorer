---
title: "Semantic Clustering Optimization for Polysemous Words"
date: 2026-01-26
category: performance-issues
tags:
  - clustering
  - kmeans
  - spatial-hashing
  - nlp
  - code-quality
  - api-design
components:
  - backend/app/embeddings/datamuse_provider.py
  - backend/app/embeddings/base.py
  - backend/app/main.py
  - src/app/explore/[word]/page.tsx
symptoms:
  - "Polysemous words showing only ~5 clusters instead of 7-12"
  - "Clustering by part-of-speech instead of semantic meaning"
  - "Duplicate KMeans clustering running twice per request"
  - "O(n²) collision avoidance causing slowdowns with 150+ words"
  - "Console.log debugging statements left in production"
  - "Missing input validation on /expand-cluster endpoint"
  - "Interface violation: expand_cluster() only on subclass"
root_cause: |
  Multiple issues compounded:
  1. Clustering used POS tags instead of GloVe embeddings
  2. Smart sampling and cluster assignment both ran KMeans independently
  3. Collision avoidance checked every pair of points naively
  4. Code review findings not addressed before merge
solution_summary: |
  Refactored clustering to use GloVe embeddings with KMeans, deduplicated
  clustering by passing precomputed assignments, optimized collision avoidance
  with spatial hashing (O(n) vs O(n²)), and addressed all code review findings.
related_docs:
  - docs/solutions/integration-issues/datamuse-api-integration-and-result-quality.md
  - PREVENTION_STRATEGIES.md
---

# Semantic Clustering Optimization for Polysemous Words

## Problem

Words like "run" have hundreds of different meanings (execute, move quickly, manage, fabric defect, sequence, etc.) but the word explorer was only showing ~5 clusters grouped by part-of-speech (noun, verb, adjective) instead of by semantic meaning.

Additionally, a code review identified several issues:
- Duplicate KMeans clustering (ran twice per request)
- O(n²) collision avoidance algorithm
- Console.log debugging left in production
- Missing input validation on API endpoints
- Interface violation (method only on subclass)

## Investigation

### Why Only 5 Clusters?

The original `_assign_clusters()` method grouped words by part-of-speech tags from Datamuse:

```python
# OLD: POS-based clustering
if "syn" in tags:
    clusters.append(0)  # Synonyms
elif first_pos == "n":
    clusters.append(1)  # Nouns
elif first_pos == "adj":
    clusters.append(2)  # Adjectives
elif first_pos == "v":
    clusters.append(3)  # Verbs
else:
    clusters.append(4)  # Other
```

This completely ignored semantic meaning - "run a program" and "run a race" would both be verbs in the same cluster.

### Why Duplicate Clustering?

The `search()` method called:
1. `_smart_sample()` - which ran KMeans to ensure diversity across senses
2. `_assign_clusters()` - which ran KMeans again for visualization

Both were clustering the same (or similar) data independently.

### Why O(n²) Collision Avoidance?

The original implementation checked every pair of points:

```python
# OLD: O(n²) - checks all pairs
for i in range(n):
    for j in range(i + 1, n):
        # Check if points are too close
```

With 150 words and 20 iterations, that's 150×150×20 = 450,000 distance checks per request.

## Solution

### 1. GloVe Embedding-Based Clustering

Use word vectors instead of POS tags:

```python
def _cluster_by_embeddings(self, words, max_clusters=12):
    # Collect embeddings for words we have vectors for
    for word in word_texts:
        if word_lower in self._word2idx:
            idx = self._word2idx[word_lower]
            valid_vectors.append(self._embeddings[idx])

    # Normalize for cosine-like clustering
    X_normalized = X / norms

    # Dynamic cluster count based on word count
    optimal_k = self._find_optimal_clusters(X_normalized)

    # KMeans clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_normalized)
```

### 2. Deduplicate Clustering

Have `_smart_sample()` return cluster assignments for reuse:

```python
def _smart_sample(self, results, limit=150, ...):
    # ... clustering logic ...

    # Return sampled results WITH their cluster assignments
    return sampled_results, sampled_assignments, cluster_label_names

def _assign_clusters(self, words, precomputed_assignments=None, ...):
    # Use precomputed if available (skip re-clustering)
    if precomputed_assignments is not None:
        return precomputed_assignments, precomputed_labels

    # Otherwise cluster from scratch
    ...
```

### 3. Spatial Hashing for O(n) Collision Avoidance

Use a grid to only check nearby points:

```python
def _avoid_collisions(self, coordinates, min_distance=0.04, iterations=20):
    cell_size = min_distance * 1.5

    for _ in range(iterations):
        # Build spatial hash grid
        grid: dict[tuple[int, int], list[int]] = {}
        for i in range(n):
            cx = int(coords[i][0] / cell_size)
            cy = int(coords[i][1] / cell_size)
            key = (cx, cy)
            if key not in grid:
                grid[key] = []
            grid[key].append(i)

        # Only check current cell and 8 neighbors
        for i in range(n):
            cx = int(coords[i][0] / cell_size)
            cy = int(coords[i][1] / cell_size)

            for dcx in (-1, 0, 1):
                for dcy in (-1, 0, 1):
                    neighbor_key = (cx + dcx, cy + dcy)
                    if neighbor_key not in grid:
                        continue
                    for j in grid[neighbor_key]:
                        if j <= i:
                            continue
                        # Check and push apart if needed
```

### 4. Input Validation

Normalize all word inputs at API boundary:

```python
@app.get("/expand-cluster")
async def expand_cluster(word: str, anchor_words: str, ...):
    # Normalize query word
    normalized_word = normalize_word(word)
    if not normalized_word:
        return JSONResponse(status_code=400, ...)

    # Normalize each anchor word
    anchors = [normalize_word(w) for w in anchor_words.split(",")]
    anchors = [w for w in anchors if w]

    # Limit anchor count
    if len(anchors) > 10:
        return JSONResponse(status_code=400, ...)
```

### 5. Interface Fix

Add default implementation to base class:

```python
class EmbeddingProvider(ABC):
    def expand_cluster(self, word, anchor_words, limit=50, exclude_words=None):
        """Expand a semantic cluster. Returns None if not supported."""
        return None  # Default: not supported
```

### 6. Remove Console.log

Cleaned up 8 debug statements from `page.tsx`:

```typescript
// REMOVED:
console.log("handleExpandCluster called with clusterId:", clusterId);
console.log("No state.data, returning early");
console.log("Anchor words for cluster", clusterId, ":", clusterWords);
// ... etc
```

## Results

- **Clusters**: 7-8 meaningful semantic clusters for "run" and "happy" instead of 5 POS-based groups
- **Performance**: Clustering runs once per request instead of twice
- **Scalability**: O(n) collision avoidance instead of O(n²)
- **Security**: All inputs validated and normalized
- **Code quality**: No debug statements, proper interface contracts

## Prevention Strategies

### Console.log in Production
- Use ESLint rule `no-console` with `allow: ['warn', 'error']`
- Add pre-commit hook to grep for `console.log`
- Configure build to strip console statements in production

### Duplicate Expensive Operations
- Document when expensive operations run
- Return intermediate results for reuse
- Add performance tests that verify operation count

### O(n²) Algorithms
- Document algorithmic complexity in function comments
- Add performance regression tests with scaling checks
- Use appropriate data structures (spatial hash, quad-tree)

### Missing Input Validation
- Validate at API boundary using schema validation
- Test invalid inputs explicitly
- Normalize all string inputs consistently

### Interface Violations
- Enable TypeScript strict mode
- Define interfaces before implementations
- Use discriminated unions for polymorphism

## Code Review Checklist

```markdown
- [ ] No console.log/debug statements
- [ ] Expensive operations run only once per request
- [ ] Algorithm complexity is documented and appropriate
- [ ] All API endpoints validate input
- [ ] No interface/type violations
- [ ] Tests cover error cases
```

## Files Changed

| File | Changes |
|------|---------|
| `backend/app/embeddings/datamuse_provider.py` | GloVe clustering, smart sampling returns assignments, spatial hashing |
| `backend/app/embeddings/base.py` | Added `expand_cluster()` to interface |
| `backend/app/main.py` | Input validation for `/expand-cluster` |
| `src/app/explore/[word]/page.tsx` | Removed 8 console.log statements |
| `src/app/api/expand-cluster/route.ts` | New API route (created) |
| `src/lib/api.ts` | Added `expandCluster()` function |
