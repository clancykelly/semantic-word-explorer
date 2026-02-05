# Handoff Document - Semantic Word Explorer

**Last updated:** 2026-02-05

## Project Overview

A semantic word exploration tool that finds related words using multiple data sources (Datamuse API, Moby Thesaurus, ConceptNet) and clusters them by meaning using GloVe embeddings.

## Current State

### Working Features

1. **Semantic search** (`/explore?word=X&mode=semantic`)
   - Fetches candidates from Datamuse API (synonyms + "means like")
   - Supplements with Moby Thesaurus and ConceptNet
   - Filters by GloVe embedding similarity
   - Clusters by semantic meaning using agglomerative clustering

2. **Contextual search** (`mode=contextual`)
   - Uses FAISS/GloVe embeddings directly
   - Requires `EMBEDDING_PROVIDER=faiss` and `EMBEDDING_DATA_DIR=data`

3. **Cluster expansion** (`/expand-cluster`)
   - Drills into a specific semantic cluster for more words

4. **Visualization**
   - Graph view with weighted links (line thickness/opacity = similarity)
   - List view with clusters, sortable by strength/formality/alphabetical
   - Font weight encodes relationship strength
   - Italic text for formal words

5. **Formality analysis**
   - Heuristic scoring based on word length, syllables, Latin/Greek suffixes
   - 0.0 = casual, 1.0 = formal

### Recent Fixes (this session)

1. **Similarity display** - Now uses GloVe relevance instead of Datamuse score (was showing 0 for synonyms)
2. **Archaic word filtering** - Words not in GloVe vocabulary are filtered out (fixed "duodecimo" for "baby")
3. **Graph links** - Added weighted lines from each word to query word

### Known Issues / Future Work

1. **ConceptNet API** - Was returning 502 errors during testing (API may be unreliable)
2. **Graph view performance** - Individual traces per word for links may be slow with many results
3. **No remote configured** - `git push` will fail, need to set up remote

## Architecture

```
Frontend (Next.js)           Backend (FastAPI)
├── /explore/[word]    →     GET /explore
├── ClusterListView          ├── DatamuseProvider (semantic)
├── WordScatterPlot          │   ├── Datamuse API
└── API routes               │   ├── Moby Thesaurus
    ├── /api/explore         │   └── ConceptNet
    └── /api/expand-cluster  └── FAISSProvider (contextual)
```

## Key Files

- `backend/app/embeddings/datamuse_provider.py` - Main semantic search logic (~1600 lines)
- `backend/app/embeddings/base.py` - WordResult dataclass, base provider
- `src/components/WordScatterPlot.tsx` - Graph visualization with Plotly
- `src/components/ClusterListView.tsx` - List view with clusters

## Running Locally

```bash
# Backend
cd backend
EMBEDDING_PROVIDER=faiss EMBEDDING_DATA_DIR=data uv run uvicorn app.main:app --reload --port 8000

# Frontend
npm run dev
```

## Data Dependencies

- GloVe embeddings in `backend/embedding_data/` (not committed, ~480MB)
- Moby Thesaurus downloaded on first use

## Documentation

- `docs/solutions/logic-errors/similarity-score-source-mismatch.md`
- `docs/solutions/logic-errors/archaic-words-bypassing-filter.md`
