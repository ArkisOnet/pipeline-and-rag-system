"""
Thread-safe async .jsonl writer.

All records are appended to a single output file with one JSON object per line.
Uses an asyncio.Lock to prevent interleaved writes from concurrent tasks.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class JsonlWriter:
    def __init__(self, output_path: str) -> None:
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._count = 0

    async def write_batch(self, records: list[dict]) -> int:
        """
        Append a list of records to the .jsonl file.
        Returns the number of records written.
        """
        if not records:
            return 0

        lines = []
        for record in records:
            try:
                lines.append(json.dumps(record, ensure_ascii=False))
            except (TypeError, ValueError) as exc:
                logger.error("Failed to serialise record: %s", exc)

        async with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
            self._count += len(lines)

        return len(lines)

    @property
    def total_written(self) -> int:
        return self._count
