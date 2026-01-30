import argparse
from lib.augmented_generation import (
    rag_command,
    summarize_command,
    citations_command,
    questions_command
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

    citations_parser = subparsers.add_parser("citations", help="Get citations for a given text")
    citations_parser.add_argument("query", type=str, help="Text to get citations for")
    citations_parser.add_argument("--limit", type=int, default=5, help="Number of documents to retrieve for citations")
    
    questions_parser = subparsers.add_parser("question", help="Get questions and answers for a given text")
    questions_parser.add_argument("question", type=str, help="Question to answer based on retrieved documents")
    questions_parser.add_argument("--limit", type=int, default=5, help="Number of documents to retrieve for context")
    
    args = parser.parse_args()

    match args.command:
        case "rag":
            # do RAG stuff here
            res = rag_command(args.query)
            print("Search Results:")
            for doc in res["search_results"]:
                print(f"- {doc['title']}")
            print("\nRAG Response:")
            print(res["answer"])
        case "summarize":
            res = summarize_command(args.query, args.limit)
            print("Search Results:")
            for doc in res["search_results"]:
                print(f"- {doc['title']}")
            print("\nLLM Summary:")
            print(res["summary"])
        case "citations":
            res = citations_command(args.query, args.limit)
            print("Search Results:")
            for doc in res["search_results"]:
                print(f"- {doc['title']}")
            print("\nLLM Answer:")
            print(res["citations"])
        case "question":
            res = questions_command(args.question, args.limit)
            print("Search Results:")
            for doc in res["search_results"]:
                print(f"- {doc['title']}")
            print("\nAnswer:")
            print(res["answer"])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()