import os
from time import sleep
from .keyword_search import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch
from .search_utils import (
    load_movies,
    format_search_result,
    DEFAULT_SEARCH_LIMIT,
    HYBRID_ALPHA,
    RRF_SEARCH_K,
)
from cli.test_gemini import client


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

    def weighted_search(self, query, alpha=HYBRID_ALPHA, limit=DEFAULT_SEARCH_LIMIT):
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

    def rrf_search(self, query, k=RRF_SEARCH_K, limit=DEFAULT_SEARCH_LIMIT):
        res_bm25 = self._bm25_search(query, limit*500)
        res_chunked = self.semantic_search.search_chunks(query, limit*500)
        
        doc_id_to_documents = {}
        
        for i, res in enumerate(res_bm25):
            doc_id = res['id']
            if doc_id not in doc_id_to_documents:
                doc_id_to_documents[doc_id] = {
                    "title": res['title'],
                    "document": res['document'],
                    "bm25_rank": i + 1,
                    "semantic_rank": None,
                    "rrf_score": rrf_score(i+1, k)
                }
            else:
                doc_id_to_documents[doc_id]['bm25_rank'] = i + 1
                doc_id_to_documents[doc_id]['rrf_score'] += rrf_score(i+1, k)
        
        for i, res in enumerate(res_chunked):
            doc_id = res['id']
            if doc_id not in doc_id_to_documents:
                doc_id_to_documents[doc_id] = {
                    "title": res['title'],
                    "document": res['document'],
                    "bm25_rank": None,
                    "semantic_rank": i + 1,
                    "rrf_score": rrf_score(i+1, k)
                }
            else:
                doc_id_to_documents[doc_id]['semantic_rank'] = i + 1
                doc_id_to_documents[doc_id]['rrf_score'] += rrf_score(i+1, k)
        
        results = []
        for doc_id, data in doc_id_to_documents.items():
            formatted_res = format_search_result(
                doc_id=doc_id,
                title=data['title'],
                document=data['document'],
                score=data['rrf_score'],
                bm25_rank= data['bm25_rank'],
                semantic_rank= data['semantic_rank'],
            )
            results.append(formatted_res)
        return sorted(results, key=lambda x: x['score'], reverse=True)[:limit]

def rrf_score(rank, k=RRF_SEARCH_K):
    return 1 / (k + rank)

def hybrid_score(bm25_score, semantic_score, alpha=HYBRID_ALPHA):
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

def run_rrf_search(query: str, k: int, limit: int):
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    results = hybrid_search.rrf_search(query, k, limit)
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['title']}")
        print(f"    RRF Score: {result['score']:.4f}")
        metadata = result.get('metadata', {})
        if 'bm25_rank' in metadata and 'semantic_rank' in metadata:
            print(f"    BM25 Rank: {metadata['bm25_rank']}, Semantic Rank: {metadata['semantic_rank']}")
        print(f"    {result['document']}")
    return results

def run_rerank_individual(results, query, k, limit):
    # Simple reranking based on individual scores
    print(f"Reranking top {limit} results using individual method...")
    print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):")
    for i, doc in enumerate(results):
        sleep(3)  # To avoid rate limiting
        rerank_prompt = f"""Rate how well this movie matches the search query.

            Query: "{query}"
            Movie: {doc.get("title", "")} - {doc.get("document", "")}

            Consider:
            - Direct relevance to query
            - User intent (what they're looking for)
            - Content appropriateness

            Rate 0-10 (10 = perfect match).
            Give me ONLY the number in your response, no other text or explanation.

            Score:"""
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=rerank_prompt
        )
        try:
            score = float(response.text.strip())
        except ValueError:
            score = 0.0
        doc['rerank_score'] = score
    new_results = sorted(results, key=lambda x: x.get('rerank_score', 0), reverse=True)
    for i, result in enumerate(new_results):
        print(f"\n{i+1}. {result['title']}")
        print(f"    Rerank Score: {result.get('rerank_score', 0):.3f}/10")
        metadata = result.get('metadata', {})
        if "bm25_rank" in metadata and "semantic_rank" in metadata:
            print(f"    BM25 Rank: {metadata['bm25_rank']}, Semantic Rank: {metadata['semantic_rank']}")
        print(f"    {result['document']}")
    return new_results