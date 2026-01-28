import argparse
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
    args = parser.parse_args()

    match args.command:
        case "normalize":
            if args.score:
                normalize_scores(args.score)
        case "weighted-search":
            run_weighted_search(args.query, args.alpha, args.limit)
        case "rrf-search":
            run_rrf_search(args.query, args.k, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()