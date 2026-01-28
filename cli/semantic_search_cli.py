#!/usr/bin/env python3

import argparse
import json
from lib.semantic_search import (
    verify_model, 
    embed_text, 
    verify_embeddings, 
    embed_query_text,
    semantic_search,
    chunk_text,
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
    
    chunk_parser = subparsers.add_parser("chunk", help="Chunk a document into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Size of each chunk")
    chunk_parser.add_argument("--overlap", type=int, default=50, help="Overlap size between chunks")
    
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
            semantic_search(args.query, args.limit)
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()