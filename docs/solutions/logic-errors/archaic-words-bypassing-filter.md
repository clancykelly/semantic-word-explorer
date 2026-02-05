# Archaic Words Bypassing Relevance Filter

---
category: logic-errors
module: datamuse_provider
tags: [moby-thesaurus, glove, filtering, vocabulary, archaic-words]
symptoms:
  - Obscure/archaic words appearing in results (e.g., "duodecimo" for "baby")
  - Words from Moby Thesaurus that don't fit the query context
  - Low-relevance words from curated sources passing filter
related_issues: []
date_solved: 2026-02-01
---

## Problem

Archaic and obscure words from Moby Thesaurus appeared in search results despite low semantic relevance. Example: "duodecimo" (a 12mo book format, associated with "small/miniature") appeared for the query "baby".

### Symptoms

- Obscure thesaurus entries in results
- Words with no GloVe embedding appearing
- Default similarity scores (0.25-0.5) allowing passage through filters

## Investigation

1. **"duodecimo" source**: Found in Moby Thesaurus under "baby" (small/miniature sense)
2. **GloVe check**: Word not in GloVe vocabulary (400K common words)
3. **Default score**: Words without embeddings get default similarity:
   - Synonyms: 0.5
   - Non-synonyms: 0.25
4. **Filter behavior**: Curated sources (Moby) had lower threshold (0.15) and these defaults passed

## Root Cause

Words from Moby Thesaurus without GloVe embeddings were assigned default similarity scores that passed the curated source threshold. The filter couldn't distinguish between "word has low similarity" and "word has no embedding (archaic/rare)".

**Location**: `backend/app/embeddings/datamuse_provider.py` in `_compute_query_relevance()` and `_filter_by_relevance()`

```python
# Default scores for words without embeddings
elif is_synonym:
    similarity = 0.5  # Trusted synonym, no embedding
else:
    similarity = 0.25  # Unknown word, benefit of doubt
```

## Solution

1. **Track embedding presence**: Add `_has_embedding` flag when computing relevance

```python
has_embedding = False
if word in self._word2idx:
    # ... compute real similarity ...
    has_embedding = True
candidate["_has_embedding"] = has_embedding
```

2. **Filter based on embedding presence**: Moby/ConceptNet words without embeddings are likely archaic

```python
if not has_embedding:
    if is_synonym:
        # Datamuse synonyms without embeddings: keep (trusted API)
        filtered.append(candidate)
    else:
        # Moby/ConceptNet without GloVe: skip (likely archaic)
        continue
```

This filters out:
- "duodecimo" (Moby, no GloVe embedding) - filtered
- "infant" (Moby, has GloVe embedding) - kept
- "newborn" (Datamuse synonym, has embedding) - kept

## Prevention

1. **Vocabulary validation**: Check if curated source words exist in primary embedding vocabulary
2. **Flag vs score**: Use explicit flags for data quality rather than overloading similarity scores
3. **Source-aware filtering**: Different sources have different reliability profiles

## Test Cases

```python
# Words that should be filtered (no GloVe embedding)
assert "duodecimo" not in search_results("baby")
assert "quodlibet" not in search_results("question")

# Words that should pass (have GloVe embedding)
assert "infant" in search_results("baby")
assert "newborn" in search_results("baby")
```

## Related

- Similarity score source mismatch (using wrong score for display)
