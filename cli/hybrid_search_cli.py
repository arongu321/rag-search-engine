import argparse
import sys
from pathlib import Path
from time import sleep

# Add parent directory to path so imports work when script is run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.hybrid_search import (
    normalize_scores,
    run_weighted_search,
    run_rrf_search,
)
from lib.reranks import (
    run_rerank_individual,
    run_rerank_batch,
    run_rerank_cross_encoder
)
from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
    HYBRID_ALPHA,
    RRF_SEARCH_K,
)
from lib.query_enhancement import enhance_query
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
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Reranking method to apply after RRF search",
    )
    args = parser.parse_args()

    match args.command:
        case "normalize":
            if args.score:
                normalize_scores(args.score)
        case "weighted-search":
            run_weighted_search(args.query, args.alpha, args.limit)
        case "rrf-search":
            if args.enhance == "rerank-method":
                results = run_rrf_search(args.query, args.k, 5*args.limit)
                if args.rerank_method == "individual":
                    run_rerank_individual(results, args.query, args.k, args.limit)
                elif args.rerank_method == "batch":
                    run_rerank_batch(results, args.query, args.k, args.limit)
                elif args.rerank_method == "cross_encoder":
                    run_rerank_cross_encoder(results, args.query, args.k, args.limit)
            elif args.enhance:
                enhanced_query = enhance_query(args.query, args.enhance, client)
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n")
                run_rrf_search(enhanced_query, args.k, args.limit)
            else:
                run_rrf_search(args.query, args.k, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()