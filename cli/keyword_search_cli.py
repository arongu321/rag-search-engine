#!/usr/bin/env python3

import argparse

from lib.keyword_search import (
    build_command, 
    search_command, 
    tf_command, 
    idf_command
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document and term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")
    
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to get IDF for")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case "tf":
            print(f"Getting term frequency for doc_id={args.doc_id}, term='{args.term}'")
            frequency = tf_command(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}': {frequency}")
        case "idf":
            print(f"Getting inverse document frequency for term='{args.term}'")
            idf_value = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf_value:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
