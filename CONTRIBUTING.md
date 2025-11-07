# Contributing to Show-Scribe

Thanks for your interest in contributing! This document describes the local development workflow and expectations for contributions.

## Getting Started

1. Install system prerequisites (Python 3.11+, FFmpeg, Git).
2. Create and activate the virtual environment:
   ```bash
   make setup
   ```
3. Run quality gates locally:
   ```bash
   make lint
   make typecheck
   make test
   ```

## Branching Strategy

- Use short-lived feature branches off `main`.
- Prefix branch names with the area of the change, e.g. `feature/pipeline-ingest`.
- Keep pull requests small and focused.

## Commit Conventions

- Follow Conventional Commits (`feat`, `fix`, `chore`, etc.).
- Reference issue IDs when available.
- Squash commits as needed before merging.

## Code Style & Quality

- Ruff handles linting with the configuration in `ruff.toml`.
- Black enforces formatting; run `make lint` before opening a PR.
- mypy enforces static typing; annotate new functions and public interfaces.
- Pytest is the canonical test runner; add coverage for new code paths.

## Pull Request Checklist

- [ ] Adds or updates tests relevant to the change.
- [ ] Updates documentation or examples if behavior changes.
- [ ] Passes `make lint`, `make typecheck`, and `make test`.
- [ ] Includes a concise summary in `CHANGELOG.md` under the _Unreleased_ section.

## Reporting Issues

File issues in GitHub with:

- A descriptive title
- Steps to reproduce
- Expected vs. actual behavior
- Logs, stack traces, or screenshots when available

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/). Be respectful, collaborative, and inclusive.
