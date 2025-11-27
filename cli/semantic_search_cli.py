#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search
def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    semantic_search_parser = subparsers.add_parser("verify", help="Verify the semantic search model")
    single_embed_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for a single text"
    )
    single_embed_parser.add_argument("text", type=str, help="Text to embed")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify the embeddings")

    embedquery_parser = subparsers.add_parser("embedquery", help="embed query")
    embedquery_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Search for a query")
    search_parser.add_argument("query", type=str, help="Query to search for")
    search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Number of results to return")




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
            res = search(args.query, args.limit)
            for i, res in enumerate(res, 1):
                print(f"{i}. {res['title']} (Score: {res['score']:.2f})")
                print(f"Description: {res['description']}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()