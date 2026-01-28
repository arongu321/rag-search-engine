import json
import os
import numpy as np


# Constants
BM25_B = 0.75
BM25_K1 = 1.5
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 0
SEMANTIC_CHUNK_MAX_SIZE = 4
SEMANTIC_SEARCH_LIMIT = 10
SCORE_PRECISION = 3

# Root project directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Helper paths
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

# Main cache directory
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

# Pickle paths
DOC_LENGTHS_PATH = os.path.join(CACHE_DIR, "doc_lengths.pkl")
DOC_MAP_PATH = os.path.join(CACHE_DIR, "doc_map.pkl")
INVERTED_INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
TERM_FREQUENCIES_PATH = os.path.join(CACHE_DIR, "term_frequencies.pkl")
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")

def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()
    
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)