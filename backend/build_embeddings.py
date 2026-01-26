#!/usr/bin/env python3
"""Build FAISS index from GloVe embeddings."""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.embeddings.faiss_provider import build_index_from_glove

if __name__ == "__main__":
    # Use 100k words for a good balance of coverage and performance
    build_index_from_glove(
        glove_path="data/glove.6B.300d.txt",
        output_dir="data/glove-6b-300d",
        max_words=100000,
    )
