"""Simple SQLite migration runner."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from ..db import SQLiteDatabase

MIGRATION_GLOB = "[0-9][0-9][0-9]_*.sql"


def discover_migration_files(directory: str | Path) -> list[Path]:
    """Return migration SQL files sorted by their numeric prefix."""
    base_path = Path(directory)
    return [
        path
        for path in sorted(base_path.glob(MIGRATION_GLOB))
        if path.is_file() and not path.name.startswith(".")
    ]


def run_migrations(
    database: SQLiteDatabase, *, migrations_dir: str | Path | None = None
) -> list[str]:
    """Apply outstanding migrations and return their filenames."""
    target_dir = Path(migrations_dir) if migrations_dir else Path(__file__).parent
    migration_files = discover_migration_files(target_dir)
    if not migration_files:
        return []

    applied: list[str] = []
    with database.connect() as connection:
        _ensure_schema_table(connection)
        existing = {
            row["filename"] for row in connection.execute("SELECT filename FROM schema_migrations;")
        }

        for migration_file in migration_files:
            if migration_file.name in existing:
                continue

            script = migration_file.read_text(encoding="utf-8")
            connection.executescript(script)
            connection.execute(
                "INSERT INTO schema_migrations (filename) VALUES (?);",
                (migration_file.name,),
            )
            applied.append(migration_file.name)

    return applied


def _ensure_schema_table(connection: Any) -> None:
    """Ensure the schema_migrations bookkeeping table exists."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL UNIQUE,
            applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
