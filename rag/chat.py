"""
Interactive CLI chat interface for the Medical RAG system.

Uses the deployed medical API endpoint at http://3.84.177.26:8000/medical-query

Run:
    python -m rag.chat [--top-k 5] [--category Кардиология]

Prereqs:
    pip install qdrant-client sentence-transformers httpx
    docker run -p 6333:6333 qdrant/qdrant
    python -m rag.indexer  (run once to populate Qdrant)
    Set API_KEY environment variable or in .env file

Commands during chat:
    /quit or /exit  — exit
    /clear          — clear conversation history
    /sources        — show sources from last answer
"""

from __future__ import annotations

import json
import logging
import os
import sys

import httpx

from config.settings import settings
from rag.retriever import Retriever, SearchResult

logger = logging.getLogger(__name__)

# Medical API configuration
MEDICAL_API_URL = "http://3.84.177.26:8000/medical-query"
MEDICAL_API_KEY = os.getenv("API_KEY", settings.get("API_KEY", ""))

_CONTEXT_TEMPLATE = """\
--- Источник {i}: {protocol} ({section}) ---
{text}
"""


def _build_context(results: list[SearchResult]) -> str:
    """Build context string from search results for the API."""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(_CONTEXT_TEMPLATE.format(
            i=i,
            protocol=r.protocol_name,
            section=r.section,
            text=r.text[:800],
        ))
    return "\n".join(parts)


def _format_sources(results: list[SearchResult]) -> str:
    """Format source citations for display."""
    lines = ["\nИсточники:"]
    for i, r in enumerate(results, 1):
        lines.append(f"  [{i}] {r.protocol_name} — {r.section}")
        lines.append(f"      {r.source_url}  (score: {r.score:.3f})")
    return "\n".join(lines)


def _call_medical_api(question: str, context: str, max_tokens: int = 1024) -> str:
    """
    Call the medical API endpoint.
    
    Args:
        question: User's medical question
        context: Context from clinical protocols
        max_tokens: Maximum tokens for response
        
    Returns:
        str: The answer from the medical assistant
        
    Raises:
        httpx.HTTPError: If API request fails
    """
    if not MEDICAL_API_KEY:
        raise ValueError(
            "API_KEY not set. Please set API_KEY environment variable or in .env file"
        )
    
    headers = {
        "Authorization": f"Bearer {MEDICAL_API_KEY}"
    }
    
    params = {
        "question": question,
        "context": context,
        "max_tokens": max_tokens,
        "language": "ru"  # Default to Russian, API will auto-detect
    }
    
    try:
        response = httpx.post(
            MEDICAL_API_URL,
            headers=headers,
            params=params,
            timeout=60.0  # 60 second timeout for model inference
        )
        response.raise_for_status()
        
        # Parse JSON response and extract answer
        data = response.json()
        return data.get("answer", data.get("text", str(data)))
        
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            raise ValueError("Invalid API_KEY. Check your authorization.")
        elif e.response.status_code == 403:
            raise ValueError("API_KEY forbidden. Check your credentials.")
        elif e.response.status_code == 503:
            raise ValueError("Medical API service unavailable. Model may be loading.")
        else:
            raise ValueError(f"API error {e.response.status_code}: {e.response.text}")
    except httpx.TimeoutException:
        raise ValueError("Request timed out. The model may be processing a large request.")
    except Exception as e:
        raise ValueError(f"Unexpected error calling medical API: {e}")


# ── Main chat loop ────────────────────────────────────────────────────────────

def chat(
    top_k: int = settings.TOP_K if hasattr(settings, 'TOP_K') else 5,
    category: str | None = None,
    max_tokens: int = 1024,
) -> None:
    """
    Run interactive chat loop with medical RAG system.
    
    Args:
        top_k: Number of search results to retrieve
        category: Optional category filter
        max_tokens: Maximum tokens for API responses
    """
    retriever = Retriever()
    filters = {"category": category} if category else None

    print(f"\n=== Medical Protocol RAG — Kazakhstan Clinical Guidelines ===")
    print(f"API Endpoint: {MEDICAL_API_URL}")
    print(f"API Key: {'✓ Set' if MEDICAL_API_KEY else '✗ NOT SET'}")
    print("Type your question in Russian or English. Commands: /quit /clear /sources")
    if category:
        print(f"Filter: category = {category}")
    print()

    last_sources: list[SearchResult] = []

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query:
            continue
            
        if query.lower() in ("/quit", "/exit"):
            print("Bye.")
            break
            
        if query.lower() == "/clear":
            last_sources.clear()
            print("Conversation cleared.")
            continue
            
        if query.lower() == "/sources":
            if last_sources:
                print(_format_sources(last_sources))
            else:
                print("No sources from previous answer.")
            continue

        # Retrieve relevant protocol sections
        try:
            results = retriever.search(query, top_k=top_k, filters=filters)
            last_sources = results
        except Exception as e:
            print(f"[Retrieval error: {e}]")
            continue

        if not results:
            print("Assistant: Релевантные протоколы не найдены. Попробуйте переформулировать вопрос.\n")
            continue

        # Build context from retrieved results
        context = _build_context(results)

        # Call medical API
        try:
            answer = _call_medical_api(
                question=query,
                context=context,
                max_tokens=max_tokens
            )
        except Exception as exc:
            print(f"[API error: {exc}]")
            continue

        # Display answer and sources
        print(f"\nAssistant: {answer}")
        print(_format_sources(results))
        print()


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical RAG Chat CLI")
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=5,
        help="Number of retrieved chunks (default: 5)"
    )
    parser.add_argument(
        "--category", 
        default=None,
        help="Filter by medical specialty, e.g. 'Кардиология'"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens for API response (default: 1024)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for medical endpoint (overrides env variable)"
    )
    
    args = parser.parse_args()

    # Override API key if provided
    if args.api_key:
        global MEDICAL_API_KEY
        MEDICAL_API_KEY = args.api_key

    # Check API key is set
    if not MEDICAL_API_KEY:
        print("ERROR: API_KEY not set!")
        print("Set it via:")
        print("  1. Environment variable: export API_KEY='your-key'")
        print("  2. Command line: python -m rag.chat --api-key 'your-key'")
        print("  3. .env file with API_KEY=your-key")
        sys.exit(1)

    logging.basicConfig(level=logging.WARNING)
    
    chat(
        top_k=args.top_k,
        category=args.category,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()