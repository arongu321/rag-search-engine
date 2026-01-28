#!/usr/bin/env python3

import argparse
import json
import string
from unittest import case
from nltk.stem import PorterStemmer
from inverted_index import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build and save inverted index")

    args = parser.parse_args()

    match args.command:
        case "build":
            # Build the inverted index
            print("Building inverted index...")
            idx = InvertedIndex()
            idx.build()
            idx.save()
            print("Inverted index built and saved to cache/")
            
            # Get documents containing 'merida' and print the first one
            docs = idx.get_documents('merida')
            if docs:
                print(f"First document for token 'merida': {docs[0]}")
        
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            with open("data/movies.json", "r") as f:
                data = json.load(f)
                movies = data["movies"]
            
            # Create translation table to remove punctuation
            translator = str.maketrans('', '', string.punctuation)
            
            # Remove punctuation from the query
            query = args.query.translate(translator)
            
            # Tokenization
            tokens = query.lower().split()
            
            # Remove stopwords
            with open("data/stopwords.txt", "r") as f:
                stopwords = set(f.read().splitlines())
            filtered_tokens = [token for token in tokens if token not in stopwords]
            
            # Remove empty tokens
            filtered_tokens = [token for token in filtered_tokens if token]
            
            # Stemming (simple suffix stripping)
            stemmer = PorterStemmer()
            stem = lambda word: stemmer.stem(word)
            
            stemmed_tokens = [stem(token) for token in filtered_tokens]

            # Dummy BM25 search implementation(if at least one token matches in title)
            results = [movie for movie in movies if any(token in movie["title"].translate(translator).lower() for token in stemmed_tokens)]
            for i, result in enumerate(sorted(results, key=lambda x: x["id"])):
                if i <= 5:
                    print(f"{i + 1}. {result['title']}")
                else:
                    break
    
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()