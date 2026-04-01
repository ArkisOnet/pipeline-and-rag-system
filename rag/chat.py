"""
Interactive CLI chat interface for the Medical RAG system.

Uses a local Ollama instance for answer generation (no API key required).
Falls back to Anthropic Claude if ANTHROPIC_API_KEY is set in .env.

Run:
    python -m rag.chat [--model llama3.2] [--top-k 5] [--category Кардиология]
    python -m rag.chat --backend anthropic --model claude-sonnet-4-6

Prereqs:
    pip install qdrant-client sentence-transformers
    docker run -p 6333:6333 qdrant/qdrant
    ollama pull llama3.2   (or any other model)
    python -m rag.indexer  (run once to populate Qdrant)

Commands during chat:
    /quit or /exit  — exit
    /clear          — clear conversation history
    /sources        — show sources from last answer
"""

from __future__ import annotations

import json
import logging
import sys

import httpx

from config.settings import settings
from rag.retriever import Retriever, SearchResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
Ты медицинский ассистент, специализирующийся на клинических протоколах Казахстана.
Отвечай на вопросы строго на основе предоставленного контекста из клинических протоколов.
Если информации недостаточно — честно сообщи об этом.
Всегда указывай источник (название протокола и раздел) при цитировании.
Отвечай на том же языке, на котором задан вопрос (русский или английский).
"""

_CONTEXT_TEMPLATE = """\
--- Источник {i}: {protocol} ({section}) ---
{text}
"""


def _build_context(results: list[SearchResult]) -> str:
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
    lines = ["\nИсточники:"]
    for i, r in enumerate(results, 1):
        lines.append(f"  [{i}] {r.protocol_name} — {r.section}")
        lines.append(f"      {r.source_url}  (score: {r.score:.3f})")
    return "\n".join(lines)


# ── LLM backends ─────────────────────────────────────────────────────────────

def _ollama_chat(messages: list[dict], model: str, ollama_url: str) -> str:
    """Send messages to Ollama /api/chat and return the assistant reply."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    resp = httpx.post(f"{ollama_url}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _anthropic_chat(messages: list[dict], model: str) -> str:
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic not installed. Run: pip install anthropic")
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


# ── Main chat loop ────────────────────────────────────────────────────────────

def chat(
    model: str = settings.OLLAMA_MODEL,
    top_k: int = settings.TOP_K,
    category: str | None = None,
    backend: str = settings.LLM_BACKEND,
    ollama_url: str = settings.OLLAMA_URL,
) -> None:
    retriever = Retriever()
    filters = {"category": category} if category else None

    print(f"\n=== Medical Protocol RAG — Kazakhstan Clinical Guidelines ===")
    print(f"Backend: {backend} | Model: {model}")
    print("Type your question in Russian or English. Commands: /quit /clear /sources")
    if category:
        print(f"Filter: category = {category}")
    print()

    # Ollama keeps system prompt as first message in history
    history: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
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
            history = [{"role": "system", "content": _SYSTEM_PROMPT}]
            last_sources.clear()
            print("Conversation cleared.")
            continue
        if query.lower() == "/sources":
            if last_sources:
                print(_format_sources(last_sources))
            else:
                print("No sources from previous answer.")
            continue

        results = retriever.search(query, top_k=top_k, filters=filters)
        last_sources = results

        if not results:
            print("Assistant: Релевантные протоколы не найдены. Попробуйте переформулировать вопрос.\n")
            continue

        context = _build_context(results)
        user_message = f"Контекст из клинических протоколов:\n\n{context}\n\nВопрос: {query}"
        history.append({"role": "user", "content": user_message})

        try:
            if backend == "ollama":
                answer = _ollama_chat(history, model=model, ollama_url=ollama_url)
            else:
                # For Anthropic, strip system message from history (passed separately)
                answer = _anthropic_chat(
                    [m for m in history if m["role"] != "system"],
                    model=model,
                )
        except Exception as exc:
            print(f"[LLM error: {exc}]")
            history.pop()  # remove unanswered user turn
            continue

        history.append({"role": "assistant", "content": answer})

        print(f"\nAssistant: {answer}")
        print(_format_sources(results))
        print()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Medical RAG Chat CLI")
    parser.add_argument("--model", default=settings.OLLAMA_MODEL,
                        help=f"LLM model name (default: {settings.OLLAMA_MODEL})")
    parser.add_argument("--top-k", type=int, default=settings.TOP_K,
                        help=f"Number of retrieved chunks (default: {settings.TOP_K})")
    parser.add_argument("--category", default=None,
                        help="Filter by medical specialty, e.g. 'Кардиология'")
    parser.add_argument("--backend", default=settings.LLM_BACKEND,
                        choices=["ollama", "anthropic"],
                        help=f"LLM backend (default: {settings.LLM_BACKEND})")
    parser.add_argument("--ollama-url", default=settings.OLLAMA_URL,
                        help=f"Ollama base URL (default: {settings.OLLAMA_URL})")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    chat(
        model=args.model,
        top_k=args.top_k,
        category=args.category,
        backend=args.backend,
        ollama_url=args.ollama_url,
    )


if __name__ == "__main__":
    main()
