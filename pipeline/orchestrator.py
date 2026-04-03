"""
Top-level pipeline orchestrator.

Run:
    python -m pipeline.orchestrator [OPTIONS]

Options:
    --content-types     Comma-separated content type IDs (default: 4)
    --specialties       Comma-separated specialty names to limit scope
    --concurrency       Max concurrent protocol fetch+process tasks (default: 3)
    --resume            Skip already-done URLs (default: True)
    --fresh             Wipe state.db and start over
    --dry-run           Process one protocol per specialty, print to stdout only
    --output            Override output .jsonl path
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import structlog

from config.settings import settings
from config.specialties import SPECIALTY_IDS
from pipeline.chunker.context_injector import inject
from pipeline.chunker.splitter import split
from pipeline.parser.html_cleaner import clean as html_clean
from pipeline.parser.markdown_converter import convert as md_convert
from pipeline.parser.metadata_extractor import extract as meta_extract
from pipeline.scraper.browser import close_browser, init_browser
from pipeline.scraper.listing_scraper import ProtocolStub, crawl_specialty
from pipeline.scraper.protocol_scraper import fetch_protocol
from pipeline.state_manager import StateManager
from pipeline.utils.rate_limiter import RateLimiter
from pipeline.writer import JsonlWriter

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)
logger = structlog.get_logger()


async def _process_protocol(
    url_row: dict,
    state: StateManager,
    writer: JsonlWriter,
    limiter: RateLimiter,
    dry_run: bool = False,
) -> None:
    url = url_row["url"]
    try:
        async with limiter:
            stub = ProtocolStub(
                url=url_row["url"],
                document_id=url_row["document_id"],
                name="",
                content_type=4,
                country="Kazakhstan",
            )
            page = await fetch_protocol(stub)

        state.set_status(url, "scraped")

        clean_html = html_clean(page.raw_html, source_url=url)
        metadata = meta_extract(page.raw_html, stub=page.stub)
        markdown = md_convert(clean_html)
        chunks = split(markdown, protocol_name=metadata.name)
        records = inject(chunks, metadata)

        if dry_run:
            for rec in records[:2]:
                print("\n--- DRY RUN CHUNK ---")
                print(rec["text"][:600])
                print("METADATA:", rec["metadata"])
            logger.info("dry_run", url=url, chunks=len(records))
        else:
            written = await writer.write_batch(records)
            logger.info("protocol_done", url=url, chunks=written, total=writer.total_written)

        state.set_status(url, "done")

    except Exception as exc:
        logger.error("protocol_failed", url=url, error=str(exc))
        state.set_status(url, "failed", error=str(exc), increment_retry=True)


async def run(
    content_types: list[int],
    specialty_filter: list[str] | None,
    concurrency: int,
    resume: bool,
    fresh: bool,
    dry_run: bool,
    output_path: str,
) -> None:
    state = StateManager(settings.STATE_DB_PATH)
    writer = JsonlWriter(output_path)
    limiter = RateLimiter(settings.REQUESTS_PER_SECOND)

    if fresh:
        logger.warning("Wiping state database (--fresh)")
        state.wipe()

    await init_browser(block_resources=True)
    try:
        # ── Phase 1: Seed ────────────────────────────────────────────────────
        logger.info("phase", name="seed", specialties=len(SPECIALTY_IDS))
        specialties_to_crawl = {
            name: sid
            for name, sid in SPECIALTY_IDS.items()
            if specialty_filter is None or name in specialty_filter
        }

        for ct in content_types:
            for name, sid in specialties_to_crawl.items():
                stubs = await crawl_specialty(
                    specialty_name=name,
                    specialty_id=sid,
                    content_type=ct,
                    state_manager=state if resume else None,
                )
                added = sum(1 for s in stubs if state.add_protocol_url(s.url, s.document_id))
                logger.info("seeded", specialty=name, new_urls=added, total=len(stubs))
                if dry_run:
                    break

        logger.info("seed_complete", stats=state.get_stats())

        # ── Phase 2: Process ─────────────────────────────────────────────────
        logger.info("phase", name="process")
        sem = asyncio.Semaphore(concurrency)

        async def bounded(row: dict) -> None:
            async with sem:
                await _process_protocol(row, state, writer, limiter, dry_run=dry_run)

        batch_size = concurrency * 10
        processed = 0
        while True:
            pending = state.get_pending_urls(limit=batch_size)
            if not pending:
                break
            tasks = [asyncio.create_task(bounded(row)) for row in pending]
            await asyncio.gather(*tasks, return_exceptions=True)
            processed += len(pending)
            logger.info("batch_done", processed=processed, stats=state.get_stats())
            if dry_run:
                break

    finally:
        await close_browser()

    logger.info(
        "pipeline_complete",
        total_written=writer.total_written,
        output=output_path,
        stats=state.get_stats(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MedElement Kazakhstan Protocol RAG Pipeline")
    parser.add_argument("--content-types", default="4")
    parser.add_argument("--specialties", default=None)
    parser.add_argument("--concurrency", type=int, default=settings.CONCURRENCY)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default=settings.OUTPUT_JSONL_PATH)
    args = parser.parse_args()

    content_types = [int(x.strip()) for x in args.content_types.split(",")]
    specialty_filter = (
        [s.strip() for s in args.specialties.split(",")]
        if args.specialties else None
    )

    asyncio.run(run(
        content_types=content_types,
        specialty_filter=specialty_filter,
        concurrency=args.concurrency,
        resume=args.resume,
        fresh=args.fresh,
        dry_run=args.dry_run,
        output_path=args.output,
    ))

    from rag.indexer import main as indexer_main

    indexer_main()


if __name__ == "__main__":
    main()
