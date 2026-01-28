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
    semantic_chunking
)
from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT, 
    DEFAULT_CHUNK_SIZE, 
    DEFAULT_CHUNK_OVERLAP,
    SEMANTIC_CHUNK_MAX_SIZE,
    SEMANTIC_SEARCH_LIMIT
)
from lib.chunked_semantic_search import (
    embed_chunks,
    search_chunks
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
    search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of top results to return")
    
    chunk_parser = subparsers.add_parser("chunk", help="Chunk a document into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Size of each chunk")
    chunk_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Overlap size between chunks")
    
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk a document semantically (not implemented)")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk semantically")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=SEMANTIC_CHUNK_MAX_SIZE, help="Maximum size of each semantic chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Overlap size between semantic chunks")
    
    subparsers.add_parser("embed_chunks", help="Embed chunks of all documents")
    
    search_chunks_parser = subparsers.add_parser("search_chunked", help="Search for similar documents using chunked semantic search")
    search_chunks_parser.add_argument("query", type=str, help="Query text to search for")
    search_chunks_parser.add_argument("--limit", type=int, default=SEMANTIC_SEARCH_LIMIT, help="Number of top results to return")
    
    
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
        case "semantic_chunk":
            semantic_chunking(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks()
        case "search_chunked":
            search_chunks(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()