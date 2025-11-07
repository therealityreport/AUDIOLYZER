"""SQLite database helpers for the Show-Scribe storage layer."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

__all__ = [
    "DatabaseError",
    "DatabaseIntegrityError",
    "SQLiteDatabase",
]


class DatabaseError(RuntimeError):
    """Raised when database operations fail."""


class DatabaseIntegrityError(DatabaseError):
    """Raised when the SQLite integrity checks fail."""


class SQLiteDatabase:
    """Lightweight helper around SQLite connections and schema management."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self, *, read_only: bool = False) -> Iterator[sqlite3.Connection]:
        """Context manager that yields a SQLite connection with sane defaults."""
        uri = self._build_uri(read_only=read_only)
        connection = sqlite3.connect(
            uri,
            uri=True,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON;")

        try:
            yield connection
            if not read_only:
                connection.commit()
        except Exception as exc:  # pragma: no cover - defensive
            if not read_only:
                connection.rollback()
            raise DatabaseError(str(exc)) from exc
        finally:
            connection.close()

    def initialize(self, schema_path: str | Path | None = None) -> None:
        """Create the database file and apply the base schema."""
        schema_file = Path(schema_path) if schema_path else Path(__file__).with_name("schema.sql")
        if not schema_file.exists():
            raise DatabaseError(f"Schema file not found: {schema_file}")

        schema_sql = schema_file.read_text(encoding="utf-8")
        with self.connect() as connection:
            connection.executescript(schema_sql)

    def executescript(self, script: str) -> None:
        """Execute a multi-statement SQL script."""
        with self.connect() as connection:
            connection.executescript(script)

    def execute(self, sql: str, parameters: Sequence[Any] | None = None) -> int:
        """Execute a modifying SQL statement and return the affected row count."""
        with self.connect() as connection:
            cursor = connection.execute(sql, parameters or [])
            return cursor.rowcount

    def fetch_one(self, sql: str, parameters: Sequence[Any] | None = None) -> sqlite3.Row | None:
        """Execute a SELECT and return a single row."""
        with self.connect(read_only=True) as connection:
            cursor = connection.execute(sql, parameters or [])
            row = cursor.fetchone()
            return cast(sqlite3.Row | None, row)

    def fetch_all(self, sql: str, parameters: Sequence[Any] | None = None) -> list[sqlite3.Row]:
        """Execute a SELECT and return all rows."""
        with self.connect(read_only=True) as connection:
            cursor = connection.execute(sql, parameters or [])
            rows = cursor.fetchall()
            return cast(list[sqlite3.Row], rows)

    def iterate(self, sql: str, parameters: Sequence[Any] | None = None) -> Iterator[sqlite3.Row]:
        """Yield rows lazily for streaming SELECT statements."""
        with self.connect(read_only=True) as connection:
            cursor = connection.execute(sql, parameters or [])
            yield from cursor

    def run_migrations(self, migrations_dir: str | Path | None = None) -> list[str]:
        """Apply outstanding migrations and return the filenames that were applied."""
        from .migrations import run_migrations  # Local import to avoid cycles

        return run_migrations(self, migrations_dir=migrations_dir)

    def check_integrity(self) -> None:
        """Run SQLite integrity and foreign key checks."""
        with self.connect(read_only=True) as connection:
            integrity = connection.execute("PRAGMA integrity_check;").fetchone()
            if not integrity or integrity[0] != "ok":
                raise DatabaseIntegrityError(f"Integrity check failed: {integrity!r}")

            fk_issues = list(connection.execute("PRAGMA foreign_key_check;"))
            if fk_issues:
                formatted = ", ".join(str(issue) for issue in fk_issues)
                raise DatabaseIntegrityError(f"Foreign key violations detected: {formatted}")

    def vacuum(self) -> None:
        """Run VACUUM to compact the database (requires exclusive lock)."""
        with self.connect() as connection:
            connection.execute("VACUUM;")

    def _build_uri(self, *, read_only: bool) -> str:
        """Build the SQLite URI for connections."""
        mode = "ro" if read_only else "rwc"
        return f"file:{self.db_path}?mode={mode}"
