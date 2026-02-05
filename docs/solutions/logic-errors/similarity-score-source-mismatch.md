# Similarity Score Source Mismatch

---
category: logic-errors
module: datamuse_provider
tags: [similarity, glove, datamuse, ui-display, font-weight]
symptoms:
  - Words appear with inconsistent bolding in list view
  - Some synonyms show no visual weight despite being relevant
  - similarity=0 for words that should have high relevance
related_issues: []
date_solved: 2026-02-01
---

## Problem

Words in the ClusterListView displayed inconsistent font weights. Some synonyms appeared un-bolded (thin) despite being highly relevant, while others displayed correctly.

### Symptoms

- Words from Datamuse `rel_syn` endpoint had `similarity=0`
- Font weight calculation `400 + sim * 300` produced weight=400 (thin) for these words
- Visual disconnect between word relevance and display weight

## Investigation

Traced the data flow:

1. **Frontend** uses `similarity` field to calculate font weight
2. **Backend** was setting `similarity = score / max_score` where `score` is the Datamuse API score
3. **Datamuse synonyms** from `rel_syn` endpoint return score=0 (they're listed as synonyms without numerical ranking)
4. **GloVe relevance** (`_relevance`) was being computed but not used for display

## Root Cause

The `similarity` field in `WordResult` was derived from Datamuse's API score, which is 0 for explicit synonyms. The GloVe-based `_relevance` score (which measures actual semantic similarity) was computed for filtering but discarded for display.

**Location**: `backend/app/embeddings/datamuse_provider.py` lines 1572-1577

```python
# BEFORE (incorrect)
score = result.get("score", 0)
similarity = score / max_score if max_score > 0 else 0
```

## Solution

Use the GloVe `_relevance` score for the `similarity` field instead of Datamuse's score:

```python
# AFTER (correct)
relevance = result.get("_relevance", 0.5)
similarity = min(1.0, relevance)
```

This ensures:
- All words have meaningful similarity values (0.5-0.7 range typical)
- Font weights reflect actual semantic relevance
- Synonyms display with appropriate visual weight

## Prevention

1. **Use semantic scores for display**: When multiple scoring systems exist (API scores vs embedding similarity), use the one that represents the concept being visualized
2. **Test with edge cases**: Datamuse synonyms are a common case that should be tested
3. **Document score sources**: Add comments explaining where scores originate

## Related

- Archaic word filtering issue (words not in GloVe vocabulary)
