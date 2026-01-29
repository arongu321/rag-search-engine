from time import sleep
import json
from cli.test_gemini import client
from sentence_transformers import CrossEncoder

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
    
    print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):")
    # Print the reranked results
    for i, result in enumerate(reranked_results[:limit]):
        print(f"\n{i+1}. {result['title']}")
        print(f"    Rerank Rank: {result.get('rerank_rank', i+1)}")
        print(f"    RRF Score: {result['score']:.4f}")
        metadata = result.get('metadata', {})
        if 'bm25_rank' in metadata and 'semantic_rank' in metadata:
            print(f"    BM25 Rank: {metadata['bm25_rank']}, Semantic Rank: {metadata['semantic_rank']}")
        print(f"    {result['document']}")
    
    return reranked_results[:limit]

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
    
    print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):")
    for i, result in enumerate(reranked_results[:limit]):
        print(f"\n{i+1}. {result['title']}")
        print(f"    Cross-Encoder Score: {result.get('cross_encoder_score', 0):.4f}")
        print(f"    RRF Score: {result['score']:.4f}")
        metadata = result.get('metadata', {})
        if 'bm25_rank' in metadata and 'semantic_rank' in metadata:
            print(f"    BM25 Rank: {metadata['bm25_rank']}, Semantic Rank: {metadata['semantic_rank']}")
        print(f"    {result['document']}")
    
    return reranked_results[:limit]