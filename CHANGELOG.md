# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Initial repository scaffolding, tooling configuration, and development environment setup.
- Speaker identification pipeline that matches diarization clusters against the voice
  bank and persists embeddings for future runs.
- Streamlit snippet labeler now supports per-snippet assignments and voice bank seeding
  via `scripts/seed_voice_bank_from_snippets.py`; added config tuning for finer diarization
  control and snippet-based seeding workflow.
- Audio preprocessing rollout documentation covering artifact contract, install options, benchmarking workflow, and merge acceptance gates.

### Changed
- Standardised on Python 3.11 across local environments, CI examples, and docs; added `.python-version` to enforce tooling alignment.
