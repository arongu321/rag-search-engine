import os
from typing import Optional
import logging

from .keyword_search import InvertedIndex
from .query_enhancement import enhance_query
from .reranking import rerank
from .search_utils import (
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
    RRF_K,
    SEARCH_MULTIPLIER,
    format_search_result,
    load_movies,
)
from .semantic_search import ChunkedSemanticSearch

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        combined = combine_search_results(bm25_results, semantic_results, alpha)
        return combined[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        fused = reciprocal_rank_fusion(bm25_results, semantic_results, k)
        return fused[:limit]


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    normalized_scores = []
    for s in scores:
        normalized_scores.append((s - min_score) / (max_score - min_score))
    return normalized_scores


def normalize_search_results(results: list[dict]) -> list[dict]:
    scores: list[float] = []
    for result in results:
        scores.append(result["score"])

    normalized: list[float] = normalize_scores(scores)
    for i, result in enumerate(results):
        result["normalized_score"] = normalized[i]

    return results


def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = DEFAULT_ALPHA
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def combine_search_results(
    bm25_results: list[dict], semantic_results: list[dict], alpha: float = DEFAULT_ALPHA
) -> list[dict]:
    bm25_normalized = normalize_search_results(bm25_results)
    semantic_normalized = normalize_search_results(semantic_results)

    combined_scores = {}

    for result in bm25_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["bm25_score"]:
            combined_scores[doc_id]["bm25_score"] = result["normalized_score"]

    for result in semantic_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["semantic_score"]:
            combined_scores[doc_id]["semantic_score"] = result["normalized_score"]

    hybrid_results = []
    for doc_id, data in combined_scores.items():
        score_value = hybrid_score(data["bm25_score"], data["semantic_score"], alpha)
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=score_value,
            bm25_score=data["bm25_score"],
            semantic_score=data["semantic_score"],
        )
        hybrid_results.append(result)

    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)


def rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1 / (k + rank)


def reciprocal_rank_fusion(
    bm25_results: list[dict], semantic_results: list[dict], k: int = RRF_K
) -> list[dict]:
    rrf_scores = {}

    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "rrf_score": 0.0,
                "bm25_rank": None,
                "semantic_rank": None,
            }
        if rrf_scores[doc_id]["bm25_rank"] is None:
            rrf_scores[doc_id]["bm25_rank"] = rank
            rrf_scores[doc_id]["rrf_score"] += rrf_score(rank, k)

    for rank, result in enumerate(semantic_results, start=1):
        doc_id = result["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "rrf_score": 0.0,
                "bm25_rank": None,
                "semantic_rank": None,
            }
        if rrf_scores[doc_id]["semantic_rank"] is None:
            rrf_scores[doc_id]["semantic_rank"] = rank
            rrf_scores[doc_id]["rrf_score"] += rrf_score(rank, k)

    sorted_items = sorted(
        rrf_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True
    )

    rrf_results = []
    for doc_id, data in sorted_items:
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=data["rrf_score"],
            rrf_score=data["rrf_score"],
            bm25_rank=data["bm25_rank"],
            semantic_rank=data["semantic_rank"],
        )
        rrf_results.append(result)

    return rrf_results


def weighted_search_command(
    query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT
) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)

    original_query = query

    search_limit = limit
    results = searcher.weighted_search(query, alpha, search_limit)

    return {
        "original_query": original_query,
        "query": query,
        "alpha": alpha,
        "results": results,
    }


def rrf_search_command(
    query: str,
    k: int = RRF_K,
    enhance: Optional[str] = None,
    rerank_method: Optional[str] = None,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)

    # Stage 1: Log original query
    original_query = query
    logging.debug("=" * 80)
    logging.debug("STAGE 1: Original Query")
    logging.debug(f"Query: '{original_query}'")
    logging.debug("=" * 80)

    # Stage 2: Query Enhancement
    enhanced_query = None
    if enhance:
        enhanced_query = enhance_query(query, method=enhance)
        query = enhanced_query
        logging.debug("=" * 80)
        logging.debug("STAGE 2: Query Enhancement")
        logging.debug(f"Enhancement Method: {enhance}")
        logging.debug(f"Original Query: '{original_query}'")
        logging.debug(f"Enhanced Query: '{enhanced_query}'")
        logging.debug("=" * 80)

    # Stage 3: RRF Search
    search_limit = limit * SEARCH_MULTIPLIER if rerank_method else limit
    results = searcher.rrf_search(query, k, search_limit)
    
    logging.debug("=" * 80)
    logging.debug("STAGE 3: RRF Search Results")
    logging.debug(f"Search Query: '{query}'")
    logging.debug(f"K Parameter: {k}")
    logging.debug(f"Number of Results: {len(results)}")
    logging.debug("Top 5 Results:")
    for i, result in enumerate(results[:5], 1):
        logging.debug(f"  {i}. {result['title']} (RRF Score: {result.get('score', 0):.4f})")
        metadata = result.get('metadata', {})
        if metadata.get('bm25_rank') or metadata.get('semantic_rank'):
            logging.debug(f"     BM25 Rank: {metadata.get('bm25_rank', 'N/A')}, "
                         f"Semantic Rank: {metadata.get('semantic_rank', 'N/A')}")
    logging.debug("=" * 80)

    # Stage 4: Re-ranking
    reranked = False
    if rerank_method:
        logging.debug("=" * 80)
        logging.debug("STAGE 4: Re-ranking")
        logging.debug(f"Re-ranking Method: {rerank_method}")
        logging.debug(f"Re-ranking {len(results)} results to top {limit}")
        
        results = rerank(query, results, method=rerank_method, limit=limit)
        reranked = True
        
        logging.debug(f"Final Results After Re-ranking ({len(results)} results):")
        for i, result in enumerate(results, 1):
            score_info = []
            if 'individual_score' in result:
                score_info.append(f"Rerank: {result['individual_score']:.2f}/10")
            if 'batch_rank' in result:
                score_info.append(f"Batch Rank: {result['batch_rank']}")
            if 'crossencoder_score' in result:
                score_info.append(f"CrossEncoder: {result['crossencoder_score']:.4f}")
            score_str = ", ".join(score_info) if score_info else f"RRF: {result.get('score', 0):.4f}"
            logging.debug(f"  {i}. {result['title']} ({score_str})")
        logging.debug("=" * 80)

    return {
        "original_query": original_query,
        "enhanced_query": enhanced_query,
        "enhance_method": enhance,
        "query": query,
        "k": k,
        "rerank_method": rerank_method,
        "reranked": reranked,
        "results": results,
    }
