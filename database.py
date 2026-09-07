"""
Persistence layer for VisionQC analysis results.

Environment note
-----------------
The original spec called for SQLAlchemy + SQLite. This build environment
has no outbound network access (pip/npm/apt are all blocked by
organization policy), so SQLAlchemy could not be installed or verified
here. Rather than ship untested code, this module implements the same
responsibilities - schema management, connection handling, and typed
CRUD access to a single ``analyses`` table - directly on top of Python's
built-in ``sqlite3`` module, wrapped in a small repository class so the
rest of the application never touches raw SQL. Swapping this module for a
SQLAlchemy `Session`/`Base` implementation later is a drop-in change: the
public surface (`Database.save_analysis`, `.get_analysis`, `.list_analyses`,
`.delete_analysis`) would stay identical.

Every analysis record stores four JSON blobs (issues, probabilities,
statistics, explainability) alongside a handful of scalar columns used for
fast listing/filtering in the history view.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    filename            TEXT NOT NULL,
    predicted_class     TEXT NOT NULL,
    quality_score       REAL NOT NULL,
    quality_label       TEXT NOT NULL,
    issues              TEXT NOT NULL,
    probabilities       TEXT NOT NULL,
    statistics          TEXT NOT NULL,
    explainability       TEXT NOT NULL,
    model_version       TEXT NOT NULL,
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses (created_at DESC);
"""


@dataclass
class AnalysisRecord:
    """A single stored analysis, matching the ``analyses`` table."""

    filename: str
    predicted_class: str
    quality_score: float
    quality_label: str
    issues: list
    probabilities: dict
    statistics: dict
    explainability: dict
    model_version: str
    id: Optional[int] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "predicted_class": self.predicted_class,
            "quality_score": self.quality_score,
            "quality_label": self.quality_label,
            "issues": self.issues,
            "probabilities": self.probabilities,
            "statistics": self.statistics,
            "explainability": self.explainability,
            "model_version": self.model_version,
            "created_at": self.created_at,
        }

    def to_summary_dict(self) -> dict:
        """Lightweight representation used by the history list endpoint."""
        return {
            "id": self.id,
            "filename": self.filename,
            "predicted_class": self.predicted_class,
            "quality_score": self.quality_score,
            "quality_label": self.quality_label,
            "model_version": self.model_version,
            "created_at": self.created_at,
        }


class Database:
    """Thread-safe SQLite repository for :class:`AnalysisRecord` objects."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else settings.DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(SCHEMA)

    def save_analysis(self, record: AnalysisRecord) -> AnalysisRecord:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO analyses (
                    filename, predicted_class, quality_score, quality_label,
                    issues, probabilities, statistics, explainability,
                    model_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.filename,
                    record.predicted_class,
                    record.quality_score,
                    record.quality_label,
                    json.dumps(record.issues),
                    json.dumps(record.probabilities),
                    json.dumps(record.statistics),
                    json.dumps(record.explainability),
                    record.model_version,
                    record.created_at,
                ),
            )
            record.id = cursor.lastrowid
        return record

    def get_analysis(self, analysis_id: int) -> Optional[AnalysisRecord]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list_analyses(self, limit: int = 50, offset: int = 0) -> list[AnalysisRecord]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM analyses ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def count_analyses(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) as c FROM analyses").fetchone()
        return int(row["c"])

    def delete_analysis(self, analysis_id: int) -> bool:
        with self._lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> AnalysisRecord:
        return AnalysisRecord(
            id=row["id"],
            filename=row["filename"],
            predicted_class=row["predicted_class"],
            quality_score=row["quality_score"],
            quality_label=row["quality_label"],
            issues=json.loads(row["issues"]),
            probabilities=json.loads(row["probabilities"]),
            statistics=json.loads(row["statistics"]),
            explainability=json.loads(row["explainability"]),
            model_version=row["model_version"],
            created_at=row["created_at"],
        )


_db_instance: Optional[Database] = None
_db_lock = threading.Lock()


def get_db() -> Database:
    """Return the process-wide singleton :class:`Database` instance."""
    global _db_instance
    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:
                _db_instance = Database()
    return _db_instance
