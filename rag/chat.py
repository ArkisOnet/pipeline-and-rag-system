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
from rag.prompts import SYSTEM_PROMPT
from rag.retriever import Retriever, SearchResult, extract_icd_codes

logger = logging.getLogger(__name__)

# Medical API configuration
MEDICAL_API_URL = "http://3.84.177.26:8000/medical-query"
MEDICAL_API_KEY = settings.API_KEY or os.getenv("API_KEY", "")

_PROTOCOL_TEMPLATE = "--- Протокол {i}: {protocol} ({section}) ---\n{text}"


def _build_protocol_context(results: list[SearchResult]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(_PROTOCOL_TEMPLATE.format(
            i=i,
            protocol=r.protocol_name,
            section=r.section,
            text=r.text[:300],
        ))
    return "\n\n".join(parts)


def _build_service_context(results: list[SearchResult]) -> str:
    """Compact text sent to the LLM — lets it reference services without reproducing a table."""
    lines = []
    for i, r in enumerate(results, 1):
        m = r.metadata
        lines.append(
            f"{i}. {m.get('service_name', '—')} "
            f"(МКБ: {m.get('icd_code', '—')}, "
            f"цена: {m.get('price', '—')}, "
            f"место: {m.get('place_of_service', '—')})"
        )
    return "\n".join(lines)


def _render_service_table(results: list[SearchResult]) -> str:
    """Clean GFM Markdown table shown directly to the user after the LLM answer."""
    header = "| № | Услуга | Код услуги | МКБ-10 | Диагноз | Цена | Место оказания |"
    sep    = "|---|--------|------------|--------|---------|------|----------------|"
    rows = []
    for i, r in enumerate(results, 1):
        m = r.metadata
        rows.append(
            f"| {i} "
            f"| {m.get('service_name', '—')} "
            f"| {m.get('service_code', '—')} "
            f"| {m.get('icd_code', '—')} "
            f"| {m.get('diagnosis', '—')} "
            f"| {m.get('price', '—')} "
            f"| {m.get('place_of_service', '—')} |"
        )
    return "\n".join([header, sep] + rows)


def _format_sources(protocol_results: list[SearchResult], service_results: list[SearchResult]) -> str:
    """Single source block appended once at the end of each response."""
    lines = ["\n---\n**Источники:**"]
    if protocol_results:
        lines.append("*Клинические протоколы МЗ РК*")
        for i, r in enumerate(protocol_results, 1):
            line = f"  [{i}] {r.protocol_name} — {r.section}"
            if r.source_url:
                line += f"  <{r.source_url}>"
            lines.append(line)
    if service_results:
        lines.append("*Реестр медицинских услуг*")
        lines.append(f"  {len(service_results)} запись(-ей) из базы услуг (таблица выше)")
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
    
    body = {
        "question": question,
        "context": context,
        "system_prompt": SYSTEM_PROMPT,
        "max_tokens": max_tokens,
        "language": "ru",
    }

    try:
        response = httpx.post(
            MEDICAL_API_URL,
            headers=headers,
            json=body,
            timeout=60.0,
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


def _get_response(question: str, top_k: int, max_tokens: int) -> str:
    retriever = Retriever()
    try: 
        results = retriever.search(question, top_k=top_k)
        context = _build_service_context(results)

        answer = _call_medical_api(question, context, max_tokens=max_tokens)

        return answer, results
    except Exception as e:
        logger.error(f"Error in _get_response: {e}")
        raise
        

# ── Main chat loop ────────────────────────────────────────────────────────────

def _try_service_retriever() -> Retriever | None:
    """Return a Retriever for the services collection, or None if not indexed yet."""
    try:
        r = Retriever(
            collection=settings.SERVICES_COLLECTION,
            icd_field="icd_code",   # services store a single ICD string
        )
        r._client.get_collection(settings.SERVICES_COLLECTION)
        return r
    except Exception:
        return None


def chat(
    top_k: int = settings.TOP_K,
    category: str | None = None,
    max_tokens: int = 1024,
) -> None:
    # Protocols use icd_codes (array); services use icd_code (string)
    protocol_retriever = Retriever(icd_field="icd_codes")
    service_retriever = _try_service_retriever()
    protocol_filters = {"category": category} if category else None

    print("\n=== Medical Protocol RAG — Kazakhstan Clinical Guidelines ===")
    print(f"API Endpoint: {MEDICAL_API_URL}")
    print(f"API Key: {'✓ Set' if MEDICAL_API_KEY else '✗ NOT SET'}")
    print(f"Services DB: {'✓ Ready' if service_retriever else '✗ Not indexed (run python -m rag.indexer_services)'}")
    print("Commands: /quit  /clear  /sources\n")
    if category:
        print(f"Filter: category = {category}")

    last_protocol_results: list[SearchResult] = []
    last_service_results: list[SearchResult] = []

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
            last_protocol_results.clear()
            last_service_results.clear()
            print("Conversation cleared.")
            continue
        if query.lower() == "/sources":
            src = _format_sources(last_protocol_results, last_service_results)
            print(src if src else "No sources from previous answer.")
            continue

        # Detect ICD-10 codes in the query for exact-match priority
        icd_codes = extract_icd_codes(query)

        # 1. Retrieve from clinical protocols (exact ICD match + semantic)
        try:
            protocol_results = protocol_retriever.search(
                query, top_k=top_k, filters=protocol_filters, icd_codes=icd_codes,
            )
            last_protocol_results = protocol_results
        except Exception as exc:
            print(f"[Protocol retrieval error: {exc}]")
            continue

        # 2. Retrieve from services DB (best-effort)
        service_results: list[SearchResult] = []
        if service_retriever:
            try:
                service_results = service_retriever.search(
                    query, top_k=top_k, icd_codes=icd_codes,
                )
                last_service_results = service_results
            except Exception as exc:
                logger.warning("Service retrieval failed: %s", exc)

        # 3. No results — give specific message for ICD queries
        if not protocol_results and not service_results:
            if icd_codes:
                print(f"Assistant: К сожалению, в базе нет данных по коду МКБ-10: {', '.join(icd_codes)}\n")
            else:
                print("Assistant: Релевантные данные не найдены. Попробуйте переформулировать вопрос.\n")
            continue

        # 4. Build combined context
        context_parts = []
        if protocol_results:
            context_parts.append("=== Клинические протоколы ===\n" + _build_protocol_context(protocol_results))
        if service_results:
            context_parts.append("=== Реестр медицинских услуг ===\n" + _build_service_context(service_results))
        context = "\n\n".join(context_parts)

        # 5. Call medical API
        try:
            answer = _call_medical_api(question=query, context=context, max_tokens=max_tokens)
        except Exception as exc:
            print(f"[API error: {exc}]")
            continue

        # Output: answer → service table → sources (each appears exactly once)
        print(f"\nAssistant: {answer}")
        if service_results:
            print(f"\n**Реестр медицинских услуг:**\n{_render_service_table(service_results)}")
        print(_format_sources(protocol_results, service_results))
        print()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Medical RAG Chat CLI")
    parser.add_argument("--top-k", type=int, default=settings.TOP_K,
                        help="Number of retrieved chunks per source (default: %(default)s)")
    parser.add_argument("--category", default=None,
                        help="Filter protocols by specialty, e.g. 'Кардиология'")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Maximum tokens for API response (default: 1024)")
    parser.add_argument("--api-key", default=None,
                        help="API key for medical endpoint (overrides .env)")
    args = parser.parse_args()

    if args.api_key:
        global MEDICAL_API_KEY
        MEDICAL_API_KEY = args.api_key

    if not MEDICAL_API_KEY:
        print("ERROR: API_KEY not set!")
        print("  1. .env file: API_KEY=your-key")
        print("  2. CLI: python -m rag.chat --api-key 'your-key'")
        sys.exit(1)

    logging.basicConfig(level=logging.WARNING)
    chat(top_k=args.top_k, category=args.category, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()