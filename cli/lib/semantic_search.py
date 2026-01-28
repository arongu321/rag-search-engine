from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import os
from torch import embedding
from .search_utils import CACHE_DIR

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")

class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        
    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("Input text must be a non-empty string.")
        res = self.model.encode([text])
        return res[0]

    def build_embeddings(self, documents):
        self.documents = documents
        document_texts = []
        for document in documents:
            self.document_map[document['id']] = document
            document_str = f"{document['title']}: {document['description']}"
            document_texts.append(document_str)
        self.embeddings = self.model.encode(document_texts, show_progress_bar=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for document in documents:
            self.document_map[document['id']] = document
        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)
    
    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        cosine_scores = []
        for doc_embedding in self.embeddings:
            score = cosine_similarity(query_embedding, doc_embedding)
            cosine_scores.append(score)
        res_list = sorted(zip(self.documents, cosine_scores), key=lambda x: x[1], reverse=True)
        return [{"score": score, "title": doc['title'], "description": doc['description']} for doc, score in res_list[:limit]]
    
def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
    return embedding

def verify_embeddings():
    semantic_search = SemanticSearch()
    with open("data/movies.json", "r") as f:
        documents = json.load(f)["movies"]
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
    
def embed_query_text(query: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
    return embedding

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)