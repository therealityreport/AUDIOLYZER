#!/usr/bin/env python3
"""Ensure canonical cast directories stay in sync with show_config.json files.

This helper can be run anytime a new show or season is added. For each
``data/shows/<SHOW_SLUG>/show_config.json`` file it will:

- create the ``cast/season_XX`` directory for the configured season number
- (re)generate ``cast_list.json`` with the canonical spellings

By default the script performs updates in-place. Use ``--dry-run`` to preview.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHOWS_ROOT = ROOT / "data" / "shows"


@dataclass(slots=True)
class SyncResult:
    show_slug: str
    season: int
    cast_file: Path
    changed: bool


def discover_show_configs(target_shows: Iterable[str] | None = None) -> list[Path]:
    """Return a list of ``show_config.json`` files to process."""
    if not SHOWS_ROOT.exists():
        return []

    configs: list[Path] = []
    for child in sorted(SHOWS_ROOT.iterdir()):
        if not child.is_dir():
            continue
        if target_shows and child.name not in target_shows:
            continue
        candidate = child / "show_config.json"
        if candidate.exists():
            configs.append(candidate)
    return configs


def load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unable to parse JSON from {path}: {exc}") from exc


def normalise_cast_members(cast_members: Iterable[dict]) -> list[str]:
    names: list[str] = []
    for member in cast_members:
        if not isinstance(member, dict):
            continue
        name = member.get("canonical_name")
        if not name:
            continue
        normalised = str(name).strip()
        if normalised:
            names.append(normalised)
    return names


def build_cast_payload(show_config: dict) -> dict[str, object]:
    cast_members = normalise_cast_members(show_config.get("cast_members", []))
    return {
        "show_name": show_config.get("show_name"),
        "show_slug": show_config.get("show_slug"),
        "season_number": show_config.get("season_number"),
        "season_code": _format_season_code(show_config.get("season_number")),
        "canonical_cast": cast_members,
        "notes": show_config.get(
            "notes",
            "Auto-generated from show_config.json by sync_cast_configs.py.",
        ),
    }


def _format_season_code(season: int | str | None) -> str | None:
    try:
        value = int(season)
    except (TypeError, ValueError):
        return None
    return f"S{value:02d}"


def sync_show_config(path: Path, *, dry_run: bool = False) -> SyncResult:
    config = load_json(path)
    show_slug = str(config.get("show_slug") or path.parent.name)

    try:
        season = int(config.get("season_number"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Show config {path} missing valid 'season_number' field.",
        ) from exc

    cast_dir = path.parent / "cast" / f"season_{season:02d}"
    cast_dir.mkdir(parents=True, exist_ok=True)
    cast_file = cast_dir / "cast_list.json"
    payload = build_cast_payload(config)

    existing = cast_file.exists()
    current_content = None
    if existing:
        try:
            with cast_file.open("r", encoding="utf-8") as handle:
                current_content = json.load(handle)
        except json.JSONDecodeError:
            current_content = None

    changed = payload != current_content

    if changed and not dry_run:
        with cast_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    return SyncResult(
        show_slug=show_slug,
        season=season,
        cast_file=cast_file,
        changed=changed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show",
        action="append",
        help="Limit sync to specific show slug(s). Can be specified multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without writing any files.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with non-zero status if any cast list would change (implies --dry-run).",
    )
    args = parser.parse_args(argv)

    if args.check:
        args.dry_run = True

    configs = discover_show_configs(args.show)
    if not configs:
        print("No show_config.json files found under data/shows.", file=sys.stderr)
        return 0

    exit_code = 0
    updates_detected = 0
    for config_path in configs:
        try:
            result = sync_show_config(config_path, dry_run=args.dry_run)
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            exit_code = 1
            continue

        status = "UPDATED" if result.changed else "ok"
        suffix = " (dry-run)" if args.dry_run and result.changed else ""
        message = (
            f"[{status}] {result.show_slug} season {result.season:02d} -> "
            f"{result.cast_file}{suffix}"
        )
        print(message)
        if result.changed:
            updates_detected += 1

    if args.check and updates_detected > 0:
        print(
            f"Detected {updates_detected} cast list update(s). Run `make sync-cast` to regenerate.",
            file=sys.stderr,
        )
        return 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
