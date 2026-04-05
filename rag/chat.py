"""
Interactive CLI chat interface for the Medical RAG system.

Roles:
    doctor  — raw query sent to Qdrant (ICD codes preserved), strict clinical output
    patient — query cleaned to 2-3 keywords before Qdrant, plain-language output

Commands during chat:
    /doctor   — switch to doctor mode
    /patient  — switch to patient mode
    /clear    — clear last sources
    /sources  — show sources from last answer
    /quit     — exit

Run:
    python -m rag.chat [--role patient] [--category Кардиология]
"""

from __future__ import annotations

import logging
import os
import re
import sys

import httpx

from config.settings import settings
from rag.prompts import DOCTOR_PROMPT, PATIENT_SYSTEM_PROMPT, QUERY_CLEANUP_PROMPT
from rag.retriever import Retriever, SearchResult, extract_icd_codes

logger = logging.getLogger(__name__)

MEDICAL_API_URL = "http://3.84.177.26:8000/medical-query"
MEDICAL_API_KEY = settings.API_KEY or os.getenv("API_KEY", "")

_ROLES = {
    "doctor":  {"label": "Врач",    "min_score": settings.DOCTOR_MIN_SCORE},
    "patient": {"label": "Пациент", "min_score": settings.PATIENT_MIN_SCORE},
}

_PROTOCOL_TEMPLATE = "--- Протокол {i}: {protocol} ({section}) ---\n{text}"


# ── Context builders ──────────────────────────────────────────────────────────

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
    """Text for LLM — ICD codes and service IDs omitted so the model never echoes them."""
    lines = []
    for i, r in enumerate(results, 1):
        m = r.metadata
        lines.append(
            f"{i}. Услуга: {m.get('service_name', '—')} | "
            f"Цена: {m.get('price', '—')} | "
            f"Место: {m.get('place_of_service', '—')}"
        )
    return "\n".join(lines)


def _render_service_table(results: list[SearchResult]) -> str:
    """3-column GFM table rendered for doctor mode — no codes."""
    header = "| Услуга | Примерная цена | Где оказывается |"
    sep    = "|--------|---------------|-----------------|"
    rows = [
        f"| {r.metadata.get('service_name', '—')} "
        f"| {r.metadata.get('price', '—')} "
        f"| {r.metadata.get('place_of_service', '—')} |"
        for r in results
    ]
    return "\n".join([header, sep] + rows)


def _is_valid_url(url: str) -> bool:
    """Reject empty strings and garbled Cyrillic URLs like 'ащщщщ'."""
    if not url:
        return False
    try:
        url.encode("ascii")
    except UnicodeEncodeError:
        return False
    return url.startswith("http")


def _format_sources(
    protocol_results: list[SearchResult],
    service_results: list[SearchResult],
) -> str:
    lines = ["\n---\n**Источники:**"]
    if protocol_results:
        lines.append("*Клинические протоколы МЗ РК*")
        for i, r in enumerate(protocol_results, 1):
            line = f"  [{i}] {r.protocol_name} — {r.section}"
            if _is_valid_url(r.source_url):
                line += f"  <{r.source_url}>"
            lines.append(line)
    if service_results:
        lines.append("*Реестр медицинских услуг*")
        lines.append(f"  {len(service_results)} запись(-ей) из базы услуг (таблица выше)")
    return "\n".join(lines)


def _strip_source_refs(text: str) -> str:
    """
    Remove source markers from patient-mode answers so users never see
    [1], [2-3], [Protocol: ...], [Section: ...], or a trailing source block.
    """
    # Remove inline reference numbers: [1], [2], [1,2], [1-3]
    text = re.sub(r"\[\d+(?:[,\-]\d+)*\]", "", text)
    # Remove trailing [Protocol: ...] / [Section: ...] blocks and everything after
    text = re.sub(
        r"\n+\[(?:Protocol|Section)[^\]]*\].*$",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Remove any trailing sources block the LLM might have hallucinated
    # Covers: Источник, Источники, Sources, Литература, References
    text = re.sub(
        r"\n+(?:---\s*)?\*?\*?(?:Источник[иа]?|Sources?|Литература|References)\*?\*?:.*$",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return text.strip()


# ── API helpers ───────────────────────────────────────────────────────────────

def _call_medical_api(
    messages: list[dict],
    question: str,
    context: str,
    max_tokens: int = 1024,
) -> str:
    """
    POST to the medical API.
    All four root fields (messages, question, context, max_tokens) are required.
    No content field may be empty/None — causes 422.
    """
    if not MEDICAL_API_KEY:
        raise ValueError("API_KEY not set. Add API_KEY to .env")

    safe_messages = [
        {"role": m["role"], "content": m.get("content", "").strip() or " "}
        for m in messages
    ]

    payload = {
        "messages": safe_messages,
        "question": question.strip() or " ",
        "context":  context.strip() or " ",
        "max_tokens": max_tokens,
        "language": "ru",
    }

    logger.debug(
        "→ API | sys_len=%d | user=%r | question=%r",
        len(safe_messages[0]["content"]),
        safe_messages[-1]["content"][:80],
        payload["question"][:60],
    )

    try:
        resp = httpx.post(
            MEDICAL_API_URL,
            headers={"Authorization": f"Bearer {MEDICAL_API_KEY}"},
            json=payload,
            timeout=150.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("answer", data.get("text", str(data)))
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status == 401:
            raise ValueError("Invalid API_KEY.")
        if status == 403:
            raise ValueError("API_KEY forbidden.")
        if status == 503:
            raise ValueError("Medical API unavailable.")
        raise ValueError(f"API error {status}: {e.response.text}")
    except httpx.TimeoutException:
        raise ValueError("Request timed out.")
    except Exception as e:
        raise ValueError(f"Unexpected API error: {e}")


def _cleanup_query(raw_query: str) -> str:
    """Patient mode only: extract 2-3 keyword search phrase via LLM."""
    if not MEDICAL_API_KEY:
        return raw_query
    try:
        messages = [
            {"role": "system", "content": QUERY_CLEANUP_PROMPT},
            {"role": "user",   "content": raw_query},
        ]
        cleaned = _call_medical_api(messages, question=raw_query, context=" ", max_tokens=15)
        cleaned = cleaned.strip().strip('"').strip("'")
        return cleaned if (cleaned and len(cleaned) < len(raw_query)) else raw_query
    except Exception as exc:
        logger.debug("Query cleanup failed, using raw query: %s", exc)
        return raw_query


# ── Core response function ────────────────────────────────────────────────────

def _get_response(
    raw_query: str,
    role: str,
    protocol_retriever: Retriever,
    service_retriever: Retriever | None,
    top_k: int,
    protocol_filters: dict | None,
    max_tokens: int,
    context_patient: str = ""
) -> tuple[str, list[SearchResult], list[SearchResult]]:
    """
    Full pipeline: retrieve → build context → call LLM → clean output.

    Returns
    -------
    answer          : str — text to display (source refs stripped for patient)
    protocol_results: list[SearchResult]
    service_results : list[SearchResult]
    """
    # 1. Query preparation
    if len(context_patient) > 2000: 
        logger.warning("Context patient is more than 2000 characters, strip it up to 2000")
        context_patient = context_patient[:2000]
    if role == "doctor":
        search_query = raw_query            # preserve ICD codes as-is
    else:
        search_query = _cleanup_query(raw_query)  # extract 2-3 keywords
    icd_codes = extract_icd_codes(raw_query)
    min_score = _ROLES[role]["min_score"]
    # 2. Retrieve from both sources
    protocol_results = protocol_retriever.search(
        search_query, top_k=top_k,
        filters=protocol_filters,
        icd_codes=icd_codes,
        min_score=min_score,
    )

    service_results: list[SearchResult] = []
    if service_retriever:
        try:
            service_results = service_retriever.search(
                search_query, top_k=top_k,
                icd_codes=icd_codes,
                min_score=min_score,
            )
        except Exception as exc:
            logger.warning("Service retrieval failed: %s", exc)

    # 3. Build context string
    context_parts = []
    if protocol_results:
        context_parts.append("=== Клинические протоколы ===\n" + _build_protocol_context(protocol_results))
    if service_results:
        context_parts.append("=== Реестр медицинских услуг ===\n" + _build_service_context(service_results))
    if context_patient: 
        context_parts.append("=== Данны о пациенте === \n" + context_patient)
    context = "\n\n".join(context_parts)

    # 4. Build messages (role-specific)
    if role == "patient":
        messages = [
            {"role": "system", "content": PATIENT_SYSTEM_PROMPT.format(context=context)},
            {"role": "user",   "content": raw_query},
        ]
    else:
        messages = [
            {"role": "system", "content": DOCTOR_PROMPT},
            {"role": "user",   "content": f"Контекст:\n{context}\n\nВопрос: {raw_query}"},
        ]

    # 5. Call LLM — single call, result used exactly once
    answer = _call_medical_api(messages, question=raw_query, context=context, max_tokens=max_tokens)

    # 6. Patient post-processing: strip [N] refs and any hallucinated source block
    if role == "patient":
        answer = _strip_source_refs(answer)

    return answer, protocol_results, service_results


# ── Retrievers ────────────────────────────────────────────────────────────────

def _try_service_retriever() -> Retriever | None:
    try:
        r = Retriever(collection=settings.SERVICES_COLLECTION, icd_field="icd_code")
        r._client.get_collection(settings.SERVICES_COLLECTION)
        return r
    except Exception:
        return None


# ── Main chat loop ────────────────────────────────────────────────────────────

def chat(
    top_k: int = settings.TOP_K,
    category: str | None = None,
    max_tokens: int = 1024,
    initial_role: str = "doctor",
) -> None:
    protocol_retriever = Retriever(icd_field="icd_codes")
    service_retriever = _try_service_retriever()
    protocol_filters = {"category": category} if category else None
    role = initial_role if initial_role in _ROLES else "doctor"

    print("\n=== Medical Protocol RAG — Kazakhstan Clinical Guidelines ===")
    print(f"API Key: {'✓ Set' if MEDICAL_API_KEY else '✗ NOT SET'}")
    print(f"Services DB: {'✓ Ready' if service_retriever else '✗ Not indexed'}")
    print(f"Mode: {_ROLES[role]['label']}  (switch with /doctor or /patient)")
    print("Commands: /doctor  /patient  /clear  /sources  /quit\n")

    last_protocol_results: list[SearchResult] = []
    last_service_results: list[SearchResult] = []

    while True:
        label = _ROLES[role]["label"]
        try:
            raw_query = input(f"[{label}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not raw_query:
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        cmd = raw_query.lower()
        if cmd in ("/quit", "/exit"):
            print("Bye.")
            break
        if cmd == "/doctor":
            role = "doctor"
            print(f"Mode → {_ROLES[role]['label']} (threshold: {_ROLES[role]['min_score']})\n")
            continue
        if cmd == "/patient":
            role = "patient"
            print(f"Mode → {_ROLES[role]['label']} (threshold: {_ROLES[role]['min_score']})\n")
            continue
        if cmd == "/clear":
            last_protocol_results.clear()
            last_service_results.clear()
            print("Conversation cleared.\n")
            continue
        if cmd == "/sources":
            src = _format_sources(last_protocol_results, last_service_results)
            print(src if src else "No sources from previous answer.")
            continue

        # ── Get response ──────────────────────────────────────────────────────
        try:
            answer, protocol_results, service_results = _get_response(
                raw_query=raw_query,
                role=role,
                protocol_retriever=protocol_retriever,
                service_retriever=service_retriever,
                top_k=top_k,
                protocol_filters=protocol_filters,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            print(f"[Error: {exc}]\n")
            continue

        # Save for /sources command
        last_protocol_results = protocol_results
        last_service_results = service_results

        # ── No results ────────────────────────────────────────────────────────
        if not protocol_results and not service_results:
            icd_codes = extract_icd_codes(raw_query)
            if icd_codes:
                print(f"\nAssistant: К сожалению, в базе нет данных по коду МКБ-10: {', '.join(icd_codes)}\n")
            else:
                print("\nAssistant: В официальных протоколах нет точного совпадения. "
                      "Пожалуйста, обратитесь к врачу.\n")
            continue

        # ── Output (printed exactly once) ─────────────────────────────────────
        print(f"\nAssistant: {answer}")

        if role == "doctor":
            # Doctor sees the service table + full sources
            if service_results:
                print(f"\n**Реестр медицинских услуг:**\n{_render_service_table(service_results)}")
            print(_format_sources(protocol_results, service_results))
        # Patient: LLM already rendered the table inside `answer`; sources are
        # stored in last_protocol_results/last_service_results for /sources command
        # but never printed automatically.

        print()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Medical RAG Chat CLI")
    parser.add_argument("--top-k", type=int, default=settings.TOP_K)
    parser.add_argument("--category", default=None,
                        help="Filter protocols by specialty, e.g. 'Кардиология'")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--role", default="doctor", choices=["doctor", "patient"],
                        help="Starting role: doctor (default) or patient")
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
    chat(
        top_k=args.top_k,
        category=args.category,
        max_tokens=args.max_tokens,
        initial_role=args.role,
    )


if __name__ == "__main__":
    main()
