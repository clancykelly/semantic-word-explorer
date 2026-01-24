# Semantic Word Explorer - Python Backend

FastAPI backend for the Semantic Word Explorer, using FAISS for fast nearest neighbor search on word embeddings.

## Quick Start (Mock Data)

```bash
# Install dependencies
pip install fastapi uvicorn pydantic numpy python-Levenshtein

# Run the server
python run.py
```

The API will be available at http://localhost:8000

## API Endpoints

- `GET /` - Health check
- `GET /words` - List available words
- `GET /explore?word=ocean` - Find related words

## Using Real Embeddings

### 1. Install ML dependencies

```bash
pip install faiss-cpu umap-learn scikit-learn
```

### 2. Download GloVe embeddings

```bash
# Download GloVe 6B (822MB compressed)
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

### 3. Build the FAISS index

```python
from app.embeddings.faiss_provider import build_index_from_glove

build_index_from_glove(
    glove_path="glove.6B.300d.txt",
    output_dir="data/glove-6b-300d",
    max_words=100000,  # Optional: limit vocabulary size
)
```

### 4. Configure environment

```bash
export EMBEDDING_PROVIDER=faiss
export EMBEDDING_DATA_DIR=data/glove-6b-300d
```

### 5. Run with real embeddings

```bash
python run.py
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `mock` | Provider type: `mock` or `faiss` |
| `EMBEDDING_DATA_DIR` | - | Path to FAISS data (required for `faiss`) |

## Data Directory Structure (FAISS)

```
data/
├── embeddings.npy      # (N, D) float32 array of word embeddings
├── words.json          # List of words in same order as embeddings
├── word2idx.json       # Dict mapping word -> index
├── coordinates.npy     # (N, 2) float32 array of UMAP coordinates
├── clusters.json       # Cluster assignments and metadata (optional)
└── faiss.index         # FAISS index file (auto-generated)
```

## Development

```bash
# Install dev dependencies
pip install pytest httpx ruff

# Run tests
pytest

# Format code
ruff format .

# Lint
ruff check .
```
