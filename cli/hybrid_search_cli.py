import argparse
import sys
from pathlib import Path

# Add parent directory to path so imports work when script is run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.hybrid_search import (
    normalize_scores,
    run_weighted_search,
    run_rrf_search,
)
from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
    HYBRID_ALPHA,
    RRF_SEARCH_K,
)
from cli.test_gemini import client

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    normalize_parser = subparsers.add_parser("normalize", help="Normalize embeddings for hybrid search")
    normalize_parser.add_argument("score", type=float, nargs="+", help="Scores to normalize")
    
    weighted_search_parser = subparsers.add_parser("weighted-search", help="Perform weighted hybrid search (not implemented)")
    weighted_search_parser.add_argument("query", type=str, help="Query text to search for")
    weighted_search_parser.add_argument("--alpha", type=float, default=HYBRID_ALPHA, help="Weight for BM25 score")
    weighted_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of top results to return")

    
    rrf_search_parser = subparsers.add_parser("rrf-search", help="Perform RRF hybrid search (not implemented)")
    rrf_search_parser.add_argument("query", type=str, help="Query text to search for")
    rrf_search_parser.add_argument("-k", type=int, default=RRF_SEARCH_K, help="RRF parameter k")
    rrf_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of top results to return")
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite"],
        help="Query enhancement method",
    )
    args = parser.parse_args()

    match args.command:
        case "normalize":
            if args.score:
                normalize_scores(args.score)
        case "weighted-search":
            run_weighted_search(args.query, args.alpha, args.limit)
        case "rrf-search":
            if args.enhance:
                if args.enhance == "spell":
                    fix_prompt = f"""Fix any spelling errors in this movie search query.

                    Only correct obvious typos. Don't change correctly spelled words or any of the capitalization.

                    Query: "{args.query}"

                    If no errors, return the original query.
                    Corrected:"""
                elif args.enhance == "rewrite":
                    fix_prompt = f"""Rewrite this movie search query to be more specific and searchable.

                    Original: "{args.query}"

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

                    Rewritten query:"""
                response = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=fix_prompt
                )
                corrected_query = response.text.strip()
                print( f"Enhanced query ({args.enhance}): '{args.query}' -> '{corrected_query}'\n")
                run_rrf_search(corrected_query, args.k, args.limit)
            else:
                run_rrf_search(args.query, args.k, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()