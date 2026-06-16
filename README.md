# Semantic Word Explorer

A writer's thesaurus that goes beyond synonyms. Enter a word and explore the
*neighborhood* of related vocabulary — clustered by meaning, weighted by
similarity, and tagged by register — as an interactive graph or a clustered list.
Built for discovering unexpected, rare, and conceptually connected words that a
traditional synonym list misses.

## How it works

- **Frontend** — Next.js (App Router, TypeScript, Tailwind). Search a word, then
  see related words as a Plotly scatter graph or a clustered list. Click any word
  to explore onward; polysemous words offer a sense picker, and any cluster can be
  expanded for more words from that corner of meaning.
- **Backend** — FastAPI. Gathers candidate words from the
  [Datamuse API](https://www.datamuse.com/api/) (synonyms + "means like", fetched
  concurrently) and the Moby Thesaurus, then uses static word vectors to (a) score
  each candidate's relevance to the query and (b) cluster the results by meaning
  (agglomerative clustering over a vectorized cosine-distance matrix).
- **Word vectors** — [Model2Vec](https://github.com/MinishLab/model2vec)
  (`potion-base-8M`, a distilled static model that encodes *any* word on demand,
  including rare ones), with an on-disk GloVe table as an offline fallback.

## Architecture

```
Next.js (frontend)              FastAPI (backend)
  /explore/[word]      ──►        GET /explore
  graph + list views              ├─ Datamuse (rel_syn + ml, concurrent)
  sense picker                    ├─ Moby Thesaurus
  cluster expansion               └─ static vectors (Model2Vec / GloVe table)
                                       relevance scoring + agglomerative clustering
```

## Running locally

### Backend

```bash
cd backend
python -m venv .venv
.venv/bin/pip install -e ".[ml]"        # or: uv sync
.venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

The first start downloads the Model2Vec model (~30MB, cached afterward). If it
isn't available, the backend falls back to an on-disk GloVe vector table (see
`EMBEDDING_DATA_DIR`). See [`backend/README.md`](backend/README.md) for API details.

### Frontend

```bash
npm install
npm run dev        # http://localhost:3000
```

The frontend proxies to the backend at `http://127.0.0.1:8000` (override with `BACKEND_URL`).

## Configuration

| Env var | Side | Default | Purpose |
|---|---|---|---|
| `EMBEDDING_MODEL` | backend | `minishlab/potion-base-8M` | Model2Vec model id |
| `EMBEDDING_DATA_DIR` | backend | `data/glove-6b-300d` | GloVe table fallback location |
| `CORS_ORIGINS` | backend | `localhost:3000` | allowed frontend origins (comma-separated) |
| `BACKEND_URL` | frontend | `http://127.0.0.1:8000` | backend address |

## Data files

The large vector artifacts (GloVe text/`.npy`, FAISS indexes, the GloVe zip) are
**not** committed — they're git-ignored. Model2Vec fetches its model on first run;
the GloVe fallback table is generated/downloaded separately if you want it.

## License

MIT
