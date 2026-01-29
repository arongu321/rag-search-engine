"""Query enhancement utilities for search improvement."""

from typing import Literal

EnhancementType = Literal["spell", "rewrite", "expand"]


def get_enhancement_prompt(query: str, enhancement_type: EnhancementType) -> str:
    """Get the appropriate enhancement prompt for a given query and type.
    
    Args:
        query: The original search query
        enhancement_type: Type of enhancement to apply
        
    Returns:
        Formatted prompt for the LLM
    """
    prompts = {
        "spell": f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words or any of the capitalization.

Query: "{query}"

If no errors, return the original query.
Corrected:""",
        
        "rewrite": f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:""",
        
        "expand": f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""
    }
    
    return prompts[enhancement_type]


def enhance_query(query: str, enhancement_type: EnhancementType, client) -> str:
    """Enhance a query using the specified enhancement type.
    
    Args:
        query: Original query text
        enhancement_type: Type of enhancement to apply
        client: Gemini API client
        
    Returns:
        Enhanced query text
    """
    prompt = get_enhancement_prompt(query, enhancement_type)
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt
    )
    return response.text.strip()
