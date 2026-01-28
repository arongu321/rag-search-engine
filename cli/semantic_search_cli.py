#!/usr/bin/env python3

import argparse
import json
from lib.semantic_search import (
    verify_model, 
    embed_text, 
    verify_embeddings, 
    embed_query_text,
    SemanticSearch
)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("verify", help="Verify the semantic search model is loaded")
    subparsers.add_parser("verify_embeddings", help="Verify the embeddings are loaded or created")
    
    embed_parser = subparsers.add_parser("embed_text", help="Generate embedding for input text")
    embed_parser.add_argument("text", type=str, help="Text to generate embedding for")
    
    embed_query_parser = subparsers.add_parser("embedquery", help="Generate embedding for a query text")
    embed_query_parser.add_argument("query", type=str, help="Query text to generate embedding for")
    
    search_parser = subparsers.add_parser("search", help="Search for similar documents given a query")
    search_parser.add_argument("query", type=str, help="Query text to search for")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of top results to return")

    args = parser.parse_args()
    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            semantic_search = SemanticSearch()
            with open("data/movies.json", "r") as f:
                documents = json.load(f)["movies"]
            semantic_search.load_or_create_embeddings(documents)
            results = semantic_search.search(args.query, args.limit)
            for i, info in enumerate(results): 
                print(f"{i+1}. {info['title']} (score: {info['score']:.4f})\n {info['description']}\n")
                if i+1 >= args.limit:
                    break
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()