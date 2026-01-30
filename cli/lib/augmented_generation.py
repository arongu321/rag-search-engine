from .google_gemini_client import client, model
from .hybrid_search import HybridSearch
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    RRF_K,
    SEARCH_MULTIPLIER,
    load_movies,
)

def get_search_results(query, limit=5):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(
        query, k=RRF_K, limit=limit * SEARCH_MULTIPLIER
    )

    return search_results[:limit]

def generate_answer(search_results, query, limit=5):
    context = ""

    for result in search_results[:limit]:
        context += f"{result['title']}: {result['document']}\n\n"

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{context}

Provide a comprehensive answer that addresses the query:"""

    response = client.models.generate_content(model=model, contents=prompt)
    return (response.text or "").strip()


def multi_document_summary(search_results, query, limit=5):
    docs_text = ""
    for i, result in enumerate(search_results[:limit], start=1):
        docs_text += f"Document {i}: {result['title']}; {result['document']}\n\n"

    prompt = f"""Provide information useful to this query by synthesizing information from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.

Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search Results:
{docs_text}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    response = client.models.generate_content(model=model, contents=prompt)
    return (response.text or "").strip()


def rag(query, limit=DEFAULT_SEARCH_LIMIT):
    search_results = get_search_results(query, limit=limit * SEARCH_MULTIPLIER)

    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }

    answer = generate_answer(search_results, query, limit)

    return {
        "query": query,
        "search_results": search_results[:limit],
        "answer": answer,
    }

def get_citations_text(search_results, query, limit=5):
    documents = ""
    for i, result in enumerate(search_results[:limit], start=1):
        documents += f"[{i}] {result['title']}: {result['document']}\n\n"
    prompt = f"""Answer the question or provide information based on the provided documents.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

    Query: {query}

    Documents:
    {documents}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources using [1], [2], etc. format when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the documents, say "I don't have enough information"
    - Be direct and informative

    Answer:"""
    
    response = client.models.generate_content(model=model, contents=prompt)
    return (response.text or "").strip()

def get_response_from_question(search_results, question, limit=5):
    context = ""

    for result in search_results[:limit]:
        context += f"{result['title']}: {result['document']}\n\n"
        
    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

        This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Question: {question}

        Documents:
        {context}

        Instructions:
        - Answer questions directly and concisely
        - Be casual and conversational
        - Don't be cringe or hype-y
        - Talk like a normal person would in a chat conversation

        Answer:
    """
    
    response = client.models.generate_content(model=model, contents=prompt)
    return (response.text or "").strip()

def rag_command(query):
    return rag(query)


def summarize_command(query, limit=5):
    search_results = get_search_results(query, limit=limit * SEARCH_MULTIPLIER)

    if not search_results:
        return {"query": query, "error": "No results found"}

    summary = multi_document_summary(search_results, query, limit)

    return {
        "query": query,
        "summary": summary,
        "search_results": search_results[:limit],
    }
    
def citations_command(query, limit=5):
    search_results = get_search_results(query, limit=limit * SEARCH_MULTIPLIER)

    if not search_results:
        return {"query": query, "error": "No results found"}

    citations = get_citations_text(search_results, query, limit)
    
    return {
        "query": query,
        "citations": citations,
        "search_results": search_results[:limit],
    }

def questions_command(question, limit=5):
    search_results = get_search_results(question, limit=limit * SEARCH_MULTIPLIER)
    if not search_results:
        return {"question": question, "error": "No results found"}

    answer = get_response_from_question(search_results, question, limit)

    return {
        "question": question,
        "search_results": search_results[:limit],
        "answer": answer,
    }