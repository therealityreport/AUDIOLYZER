"""Database backup and restore helpers."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

__all__ = [
    "create_backup",
    "list_backups",
    "prune_backups",
    "restore_backup",
]

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def create_backup(
    database_path: str | Path,
    backup_dir: str | Path,
    *,
    keep: int | None = None,
) -> Path:
    """Copy the SQLite database to the backup directory and optionally prune old copies."""
    db_path = Path(database_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    backup_directory = Path(backup_dir)
    backup_directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime(TIMESTAMP_FORMAT)
    backup_name = f"{db_path.stem}_{timestamp}{db_path.suffix or '.sqlite3'}"
    destination = backup_directory / backup_name

    shutil.copy2(db_path, destination)

    if keep is not None:
        prune_backups(backup_directory, keep=keep)

    return destination


def list_backups(backup_dir: str | Path) -> list[Path]:
    """Return all backups sorted from newest to oldest."""
    directory = Path(backup_dir)
    if not directory.exists():
        return []

    backups = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix in {".db", ".sqlite", ".sqlite3"}
    ]
    backups.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return backups


def prune_backups(backup_dir: str | Path, *, keep: int) -> None:
    """Remove backups beyond the newest ``keep`` files."""
    backups = list_backups(backup_dir)
    for stale_backup in backups[keep:]:
        try:
            stale_backup.unlink()
        except OSError:
            # Intentionally swallow errors to avoid breaking backup routines.
            continue


def restore_backup(backup_path: str | Path, destination: str | Path) -> Path:
    """Restore a backup to the requested destination path."""
    source = Path(backup_path)
    if not source.exists():
        raise FileNotFoundError(f"Backup not found: {source}")

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target
