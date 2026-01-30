import argparse
from lib.augmented_generation import (
    rag_command,
    summarize_command,
)

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    
    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize a given text using LLM"
    )
    summarize_parser.add_argument("query", type=str, help="Text to summarize")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Number of documents to retrieve for context")
    
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            # do RAG stuff here
            res = rag_command(args.query)
            print("Search Results:")
            for doc in res["retrieved_documents"]:
                print(f"- {doc}")
            print("\nRAG Response:")
            print(res["answer"])
        case "summarize":
            res = summarize_command(args.query, args.limit)
            print("Search Results:")
            for doc in res["retrieved_documents"]:
                print(f"- {doc}")
            print("\nLLM Summary:")
            print(res["summary"])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()