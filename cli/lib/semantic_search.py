from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
import os
from torch import embedding
from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    SEMANTIC_CHUNK_MAX_SIZE,
    MOVIE_EMBEDDINGS_PATH,
    load_movies,
    cosine_similarity,
)

class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        
    def generate_embedding(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("Input text must be a non-empty string.")
        res = self.model.encode([text])
        return res[0]

    def build_embeddings(self, documents) -> np.ndarray:
        self.documents = documents
        document_texts = []
        for document in documents:
            self.document_map[document['id']] = document
            document_str = f"{document['title']}: {document['description']}"
            document_texts.append(document_str)
        self.embeddings = self.model.encode(document_texts, show_progress_bar=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for document in documents:
            self.document_map[document['id']] = document
        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)
    
    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
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
    documents = load_movies()
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

def semantic_search(query, limit=DEFAULT_SEARCH_LIMIT):
    semantic_search = SemanticSearch()
    documents = load_movies()
    semantic_search.load_or_create_embeddings(documents)
    results = semantic_search.search(query, limit)
    for i, info in enumerate(results): 
        print(f"{i+1}. {info['title']} (score: {info['score']:.4f})\n {info['description']}\n")
        
        
def fixed_size_chunking(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    words = text.split()
    chunks = []

    n_words = len(words)
    i = 0
    while i < n_words:
        chunk_words = words[i : i + chunk_size]
        if chunks and len(chunk_words) <= overlap:
            break

        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap

    return chunks

def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = fixed_size_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")
        
def semantic_chunking(
    text: str,
    max_chunk_size: int = SEMANTIC_CHUNK_MAX_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    # Strip leading and trailing whitespace from input text
    text = text.strip()
    
    if not text:
        print("Input text is empty after stripping whitespace. Returning empty chunk list.")
        return []
    
    # Split text into sentences using regex
    sentences = re.split(r"(?<=[.!?])\s+", text)
    
    # Check if there's only one sentence and it doesn't end with a punctuation mark
    if len(sentences) == 1 and not sentences[0].endswith(( '.', '!', '?' )):
        print("Input text appears to be a single sentence without proper punctuation. Using fixed-size chunking instead.")
        return fixed_size_chunking(text, chunk_size=max_chunk_size, overlap=overlap)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current_chunk) < max_chunk_size:
            current_chunk.append(sentence)
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_chunk.append(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")

    return chunks