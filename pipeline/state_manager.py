"""
SQLite-backed crawl state manager.

Tracks every discovered protocol URL with a status lifecycle:
  pending → scraped → parsed → done
                             ↘ failed

Also tracks listing pages so pagination is never re-crawled on resume.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateManager:
    def __init__(self, db_path: str = "data/state.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS protocol_urls (
                    url           TEXT PRIMARY KEY,
                    document_id   TEXT NOT NULL,
                    status        TEXT NOT NULL DEFAULT 'pending',
                    error_msg     TEXT,
                    retry_count   INTEGER NOT NULL DEFAULT 0,
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS listing_pages (
                    url        TEXT PRIMARY KEY,
                    done       INTEGER NOT NULL DEFAULT 0,
                    crawled_at TEXT
                );

                CREATE TABLE IF NOT EXISTS run_metadata (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_protocol_status
                    ON protocol_urls (status);
            """)

    # ------------------------------------------------------------------
    # Protocol URL methods
    # ------------------------------------------------------------------

    def add_protocol_url(self, url: str, document_id: str) -> bool:
        """Insert a new URL in 'pending' state. Returns True if newly inserted."""
        with self._conn() as conn:
            result = conn.execute(
                """
                INSERT OR IGNORE INTO protocol_urls
                    (url, document_id, status, created_at, updated_at)
                VALUES (?, ?, 'pending', ?, ?)
                """,
                (url, document_id, _now(), _now()),
            )
            return result.rowcount > 0

    def get_pending_urls(self, limit: int = 100) -> list[dict]:
        """Return up to `limit` pending protocol URL rows."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT url, document_id, retry_count
                FROM protocol_urls
                WHERE status = 'pending'
                ORDER BY created_at
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def set_status(
        self,
        url: str,
        status: str,
        error: str | None = None,
        increment_retry: bool = False,
    ) -> None:
        """Update the status of a protocol URL."""
        with self._conn() as conn:
            if increment_retry:
                conn.execute(
                    """
                    UPDATE protocol_urls
                    SET status = ?, error_msg = ?, retry_count = retry_count + 1,
                        updated_at = ?
                    WHERE url = ?
                    """,
                    (status, error, _now(), url),
                )
            else:
                conn.execute(
                    """
                    UPDATE protocol_urls
                    SET status = ?, error_msg = ?, updated_at = ?
                    WHERE url = ?
                    """,
                    (status, error, _now(), url),
                )

    def requeue_failed(self, max_retries: int = 3) -> int:
        """Reset failed URLs back to pending if under retry limit. Returns count."""
        with self._conn() as conn:
            result = conn.execute(
                """
                UPDATE protocol_urls
                SET status = 'pending', error_msg = NULL, updated_at = ?
                WHERE status = 'failed' AND retry_count < ?
                """,
                (_now(), max_retries),
            )
            return result.rowcount

    # ------------------------------------------------------------------
    # Listing page methods
    # ------------------------------------------------------------------

    def mark_listing_done(self, url: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO listing_pages (url, done, crawled_at)
                VALUES (?, 1, ?)
                ON CONFLICT(url) DO UPDATE SET done = 1, crawled_at = excluded.crawled_at
                """,
                (url, _now()),
            )

    def is_listing_done(self, url: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT done FROM listing_pages WHERE url = ?", (url,)
            ).fetchone()
            return bool(row and row["done"])

    # ------------------------------------------------------------------
    # Stats / metadata
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS cnt FROM protocol_urls GROUP BY status"
            ).fetchall()
            return {row["status"]: row["cnt"] for row in rows}

    def set_meta(self, key: str, value: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO run_metadata (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def get_meta(self, key: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM run_metadata WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else None

    def wipe(self) -> None:
        """Delete all state — use only with --fresh flag."""
        with self._conn() as conn:
            conn.executescript(
                "DELETE FROM protocol_urls; "
                "DELETE FROM listing_pages; "
                "DELETE FROM run_metadata;"
            )
