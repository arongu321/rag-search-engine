import os

from .keyword_search import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch
from .search_utils import (
    load_movies,
    format_search_result,
)


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        res_bm25 = self._bm25_search(query, limit*500)
        res_chunked = self.semantic_search.search_chunks(query, limit*500)
        
        # Normalize scores separately
        res_bm25 = normalize_search_results(res_bm25)
        res_chunked = normalize_search_results(res_chunked)
        
        combined_res = {}
        for res in res_bm25:
            doc_id = res['id']
            if doc_id not in combined_res:
                combined_res[doc_id] = {
                    "title": res['title'],
                    "document": res['document'],
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                }
            if res['normalized_score'] > combined_res[doc_id]['bm25_score']:
                combined_res[doc_id]['bm25_score'] = res['normalized_score']

        for res in res_chunked:
            doc_id = res['id']
            if doc_id not in combined_res:
                combined_res[doc_id] = {
                    "title": res['title'],
                    "document": res['document'],
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                }
            if res['normalized_score'] > combined_res[doc_id]['semantic_score']:
                combined_res[doc_id]['semantic_score'] = res['normalized_score']

        hybrid_res = []
        for doc_id, data in combined_res.items():
            hybrid_score_val = hybrid_score(
                data['bm25_score'],
                data['semantic_score'],
                alpha
            )
            formatted_res = format_search_result(
                doc_id=doc_id,
                title=data['title'],
                document=data['document'],
                score=hybrid_score_val,
                bm25_score=data['bm25_score'],
                semantic_score=data['semantic_score'],
            )
            hybrid_res.append(formatted_res)
        
        return sorted(hybrid_res, key=lambda x: x['score'], reverse=True)[:limit]

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    
def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    normalized_scores = []
    for score in scores:
        if max_score - min_score > 0:
            normalized_scores.append((score - min_score) / (max_score - min_score))
    return normalized_scores
    
def normalize_search_results(results):
    scores = [res['score'] for res in results]
    normalized_scores = normalize_scores(scores)
    for i, res in enumerate(results):
        res['normalized_score'] = normalized_scores[i]
    return results

def run_weighted_search(query: str, alpha: float, limit: int):
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    results = hybrid_search.weighted_search(query, alpha, limit)
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['title']}")
        print(f"    Hybrid Score: {result['score']:.4f}")
        metadata = result.get('metadata', {})
        if 'bm25_score' in metadata and 'semantic_score' in metadata:
            print(f"    BM25: {metadata['bm25_score']:.4f}, Semantic: {metadata['semantic_score']:.4f}")
        print(f"    {result['document']}")
              
    return results