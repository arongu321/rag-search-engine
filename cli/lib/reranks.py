from time import sleep
import json
from functools import wraps
from cli.test_gemini import client
from sentence_transformers import CrossEncoder


def print_reranked_results(results, query, k, limit, score_label="Rerank Score", score_key="rerank_score"):
    """Print reranked search results in a consistent format.
    
    Args:
        results: List of search results to print
        query: Original search query
        k: RRF parameter k value
        limit: Maximum number of results to display
        score_label: Label for the reranking score (e.g., "Rerank Score", "Cross-Encoder Score")
        score_key: Key in result dict for the reranking score
    """
    print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):")
    for i, result in enumerate(results[:limit]):
        print(f"\n{i+1}. {result['title']}")
        
        # Print the reranking score if present
        if score_key in result:
            score_value = result[score_key]
            # Format based on score type (some are 0-10, others are similarity scores)
            if score_key == "rerank_score":
                print(f"    {score_label}: {score_value:.3f}/10")
            else:
                print(f"    {score_label}: {score_value:.4f}")
        
        # Print rerank rank if present
        if "rerank_rank" in result:
            print(f"    Rerank Rank: {result['rerank_rank']}")
        
        # Print RRF score if present
        if "score" in result:
            print(f"    RRF Score: {result['score']:.4f}")
        
        # Print BM25 and Semantic ranks if available
        metadata = result.get('metadata', {})
        if 'bm25_rank' in metadata and 'semantic_rank' in metadata:
            print(f"    BM25 Rank: {metadata['bm25_rank']}, Semantic Rank: {metadata['semantic_rank']}")
        
        print(f"    {result['document']}")


def with_result_printing(score_label, score_key):
    """Decorator that automatically prints reranked results.
    
    Args:
        score_label: Label for the reranking score display
        score_key: Key in result dict for the reranking score
        
    Returns:
        Decorator function that wraps rerank functions
    """
    def decorator(func):
        @wraps(func)
        def wrapper(results, query, k, limit):
            # Call the original rerank function
            reranked_results = func(results, query, k, limit)
            # Automatically print the results
            print_reranked_results(reranked_results, query, k, limit, score_label, score_key)
            return reranked_results
        return wrapper
    return decorator


@with_result_printing(score_label="Rerank Score", score_key="rerank_score")
def run_rerank_individual(results, query, k, limit):
    # Simple reranking based on individual scores
    print(f"Reranking top {limit} results using individual method...")
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
    return sorted(results, key=lambda x: x.get('rerank_score', 0), reverse=True)

@with_result_printing(score_label="Rerank Rank", score_key="rerank_rank")
def run_rerank_batch(results, query, k, limit):
    print(f"Reranking top {limit} results using batch method...\n")
    
    doc_list_str = ""
    for doc in results:
        doc_list_str += f"{doc.get('id', '')}: {doc.get('title', '')}: {doc.get('document', '')}\n"
    
    rerank_prompt = f"""Rank these movies by relevance to the search query.

        Query: "{query}"

        Movies:
        {doc_list_str}

        Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

        [75, 12, 34, 2, 1]
        """
    
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=rerank_prompt
    )
    
    try:
        ranked_ids = json.loads(response.text.strip())
    except json.JSONDecodeError:
        print("Error: Could not parse LLM response as JSON")
        ranked_ids = []
    
    # Create a mapping from ID to document
    id_to_doc = {doc['id']: doc for doc in results}
    
    # Reorder results based on ranked_ids
    reranked_results = []
    for i, doc_id in enumerate(ranked_ids):
        reranked_results.append({**id_to_doc[doc_id], "rerank_rank": i+1})
    
    return reranked_results[:limit]

@with_result_printing(score_label="Cross-Encoder Score", score_key="cross_encoder_score")
def run_rerank_cross_encoder(results, query, k, limit):
    print(f"Reranking top {limit} results using cross_encoder method...\n")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    
    pairs = []
    for doc in results:
        pairs.append((query, f"{doc.get('title', '')} - {doc.get('document', '')}"))
    
    scores = cross_encoder.predict(pairs)
    
    for i, doc in enumerate(results):
        doc['cross_encoder_score'] = scores[i]
    
    reranked_results = sorted(results, key=lambda x: x.get('cross_encoder_score', 0), reverse=True)
    
    return reranked_results[:limit]