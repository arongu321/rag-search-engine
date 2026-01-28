import numpy as np
import json
import os
from .semantic_search import (
    SemanticSearch,
    semantic_chunking
)
from .search_utils import (
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    SEMANTIC_SEARCH_LIMIT,
    SCORE_PRECISION,
    load_movies,
    cosine_similarity,
    format_search_result,
)

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        
    def build_chunk_embeddings(self, documents) -> np.ndarray:
        self.documents = documents
        for document in documents:
            self.document_map[document['id']] = document
        chunks_list = []
        chunks_metadata = []
        for i, doc in enumerate(documents):
            if not doc['description'] or not doc['description'].strip():
                continue
            chunks = semantic_chunking(doc['description'], overlap=1)
            chunks_list.extend(chunks)
            chunks_metadata.extend([{'movie_idx': i ,'chunk_idx': j, 'total_chunks': len(chunks)} for j in range(len(chunks))])
        self.chunk_embeddings = self.model.encode(chunks_list, show_progress_bar=True)
        self.chunk_metadata = chunks_metadata
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        json.dump({"chunks": chunks_metadata, "total_chunks": len(chunks_list)}, open(CHUNK_METADATA_PATH, "w"), indent=2)
        return self.chunk_embeddings
        
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for document in documents:
            self.document_map[document['id']] = document
        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(CHUNK_METADATA_PATH):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, "r") as f:
                metadata = json.load(f)
            if len(self.chunk_embeddings) == metadata["total_chunks"]:
                self.chunk_metadata = metadata["chunks"]
                return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int = SEMANTIC_SEARCH_LIMIT) -> list[dict]:
        query_embedding = self.generate_embedding(query)
        chunk_scores = []
        for i, doc_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, doc_embedding)
            score_dict = {
                "chunk_idx": i,
                "movie_idx": self.chunk_metadata[i]['movie_idx'],
                "score": score,
            }
            chunk_scores.append(score_dict)
        movie_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score['movie_idx']
            if movie_idx not in movie_scores or movie_scores[movie_idx] <  chunk_score['score']:
                movie_scores[movie_idx] = chunk_score['score']
        sorted_movie_scores = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        results = []
        for movie_idx, score in sorted_movie_scores:
            if movie_idx is None:
                continue
            doc = self.documents[movie_idx]
            formatted_res = format_search_result(
                doc_id=doc['id'],
                title=doc["title"],
                document=doc["description"][:100],
                score=score,
                metadata=self.chunk_metadata[movie_idx] or {}
            )
            results.append(formatted_res)
        return results

def embed_chunks():
    documents = load_movies()
    chunked_search = ChunkedSemanticSearch()
    embeddings = chunked_search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")

def search_chunks(query: str, limit: int=SEMANTIC_SEARCH_LIMIT):
    documents = load_movies()
    chunked_search = ChunkedSemanticSearch()
    chunked_search.load_or_create_chunk_embeddings(documents)
    results = chunked_search.search_chunks(query, limit)
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['title']} (score: {result['score']:.{SCORE_PRECISION}f})")
        print(f"   {result['document']}")
        