from .search_utils import (
    load_movies
)

from .hybrid_search import HybridSearch
from .semantic_search import SemanticSearch
from .google_gemini_client import client, model

def rag_command(query: str) -> dict:
    # Step 1: Semantic Search to retrieve relevant documents
    retrieved_docs = get_rrf_search_results(query, limit=5)

    # Step 2: Generate answer using retrieved documents
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Query: {query}

        Documents:
        {retrieved_docs}

        Provide a comprehensive answer that addresses the query:
    """
    
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    answer = response.text.strip()

    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "answer": answer,
    }
    
def summarize_command(query: str, limit: int = 5) -> dict:

    # Step 1: Semantic Search to retrieve relevant documents
    retrieved_docs = get_rrf_search_results(query, limit)

    # Step 2: Generate summary using retrieved documents
    prompt = f"""
        Provide information useful to this query by synthesizing information from multiple search results in detail.
        The goal is to provide comprehensive information so that users know what their options are.
        Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
        This should be tailored to Hoopla users. Hoopla is a movie streaming service.
        Query: {query}
        Search Results:
        {retrieved_docs}
        Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
    """
    
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    summary = response.text.strip()

    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "summary": summary,
    }
    
def get_rrf_search_results(query: str, limit: int = 5) -> list[str]:
    movies = load_movies()
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    searcher = HybridSearch(movies)

    search_results = searcher.rrf_search(query, k=10, limit=limit)

    retrieved_docs = []
    for result in search_results:
        title = result.get("title", "")
        if title:
            retrieved_docs.append(title)

    return retrieved_docs