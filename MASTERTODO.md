# Show-Scribe Master TODO

**Project:** Show-Scribe Audio Analyzer
**Version:** 1.0 MVP
**Last Updated:** October 21, 2025
**Status:** Implementation Phase

---

## Table of Contents

0. [Next Actions](#next-actions)
1. [Project Setup & Infrastructure](#project-setup--infrastructure)
2. [Core Pipeline - Stage 1: Audio Extraction](#core-pipeline---stage-1-audio-extraction)
3. [Core Pipeline - Stage 2: Transcription (ASR)](#core-pipeline---stage-2-transcription-asr)
4. [Core Pipeline - Stage 3: Diarization](#core-pipeline---stage-3-diarization)
5. [Core Pipeline - Stage 3b: Bleep Detection](#core-pipeline---stage-3b-bleep-detection)
6. [Core Pipeline - Stage 4: Speaker Identification](#core-pipeline---stage-4-speaker-identification)
7. [Voice Bank System](#voice-bank-system)
8. [Transcript Generation & Export](#transcript-generation--export)
9. [Analytics & Reporting](#analytics--reporting)
10. [Checkpoint & Resume System](#checkpoint--resume-system)
11. [User Interface (Streamlit)](#user-interface-streamlit)
12. [CLI Interface](#cli-interface)
13. [Testing & Quality Assurance](#testing--quality-assurance)
14. [Documentation](#documentation)
15. [Deployment & Distribution](#deployment--distribution)
16. [Future Phases (v1.5 - v4.0)](#future-phases-v15---v40)

---

## Next Actions

- 🔴 Streamlit review console: wire "Process Episode" actions to the actual pipeline runner and status polling.
- 🟡 Continue canonical cast directory roll-out for future shows/seasons beyond RHOSLC S06.
- 🟢 Monitor the new `sync_cast_configs.py --check` pre-commit hook and adjust developer onboarding docs as needed.

---

## Legend

- 🔴 **Critical Path** - Blocks other work
- 🟡 **High Priority** - Core MVP feature
- 🟢 **Medium Priority** - Important but not blocking
- 🔵 **Low Priority** - Nice to have
- ⚠️ **Blocked** - Waiting on dependency
- ✅ **Complete** - Done
- 🚧 **In Progress** - Currently being worked on

**Time Estimates:**
- XS: <2 hours
- S: 2-4 hours
- M: 4-8 hours (1 day)
- L: 1-3 days
- XL: 3-5 days
- XXL: 1-2 weeks

---

## Project Setup & Infrastructure

### Repository Setup
- ✅ **Complete** - Done: 🔴 Initialize Git repository (XS)
- ✅ **Complete** - Done: 🔴 Create directory structure per DIRECTORY_STRUCTURE.md (S)
- ✅ **Complete** - Done: 🔴 Setup .gitignore with proper exclusions (XS)
- ✅ **Complete** - Done: 🟡 Create pyproject.toml with dependencies (M)
- ✅ **Complete** - Done: 🟡 Setup Makefile with targets (setup, dev, lint, test, run) (S)
- ✅ **Complete** - Done: 🟢 Create LICENSE file (MIT) (XS)
- ✅ **Complete** - Done: 🟢 Create CONTRIBUTING.md guidelines (S)
- ✅ **Complete** - Done: 🟢 Create CHANGELOG.md (XS)

### Development Environment
- ✅ **Complete** - Done: 🔴 Setup Python virtual environment (XS)
- ✅ **Complete** - Done: 🔴 Install core dependencies (Whisper, Pyannote, FFmpeg) (M)
- ✅ **Complete** - Done: 🔴 Create requirements.txt / pyproject.toml (S)
- ✅ **Complete** - Done: 🟡 Setup pre-commit hooks (.pre-commit-config.yaml) (S)
- ✅ **Complete** - Done: 🟡 Configure Ruff linter (ruff.toml) (XS)
- ✅ **Complete** - Done: 🟡 Configure Black formatter (pyproject.toml) (XS)
- ✅ **Complete** - Done: 🟡 Configure mypy type checker (mypy.ini) (S)
- ✅ **Complete** - Done: 🟢 Setup pytest configuration (pyproject.toml) (S)
- ✅ **Complete** - Done: 🟢 Create sample test fixtures (audio/video samples) (M)

### Configuration System
- ✅ **Complete** - Done: 🔴 Create config schema (src/show_scribe/config/schema.json) (M)
- ✅ **Complete** - Done: 🔴 Implement config loader (src/show_scribe/config/load.py) (M)
- ✅ **Complete** - Done: 🟡 Create default config files (configs/dev.yaml, configs/prod.yaml) (S)
- ✅ **Complete** - Done: 🟡 Implement environment variable overrides (S)
- ✅ **Complete** - Done: 🟢 Create .env.example template (XS)

### Name Management System ⭐ NEW
- ✅ **Complete** - Done: 🔴 Design cast_members schema in show_config.json (S)
  - ✅ **Complete** - canonical_name field
  - ✅ **Complete** - common_misspellings array
  - ✅ **Complete** - aliases array
  - ✅ **Complete** - role field
- ✅ **Complete** - Done: 🔴 Update speaker_profiles table with display_name field (S)
- ✅ **Complete** - Done: 🔴 Implement NameCorrector class (utils/name_correction.py) (L)
  - ✅ **Complete** - Normalize name function
  - ✅ **Complete** - Canonical name lookup
  - ✅ **Complete** - Alias matching
  - ✅ **Complete** - Misspelling detection
  - ✅ **Complete** - Fuzzy matching
  - ✅ **Complete** - Transcript correction function
- ✅ **Complete** - Done: 🟡 Create name selector UI component (ui/streamlit_app/components/name_selector.py) (M)
  - ✅ **Complete** - Dropdown with canonical names
  - ✅ **Complete** - Auto-complete functionality
  - ✅ **Complete** - "Other" option with fuzzy suggestions
- ✅ **Complete** - Done: 🟡 Integrate name correction into transcript builder (M)
- ✅ **Complete** - Done: 🟢 Add name correction logging (S)
- ✅ **Complete** - Done: 🟢 Create test suite for name correction (M)
- ✅ **Complete** - Done: 🟡 Create canonical cast directory for RHOBH S05 (data/shows/RHOBH/cast/season_05/cast_list.json) (S)
- ✅ **Complete** - Done: 🟡 Publish RHOBH S05 show_config.json with misspelling map (data/shows/RHOBH/show_config.json) (S)
- ✅ **Complete** - Done: 🟡 Create canonical cast directory for RHOSLC S06 (data/shows/RHOSLC/cast/season_06/cast_list.json) (S)
- ✅ **Complete** - Done: 🟡 Publish RHOSLC S06 show_config.json with misspelling map (data/shows/RHOSLC/show_config.json) (S)
- 🟡 Continue canonical cast directory roll-out for future shows/seasons (M)
- ✅ **Complete** - Done: 🟢 Add sync_cast_configs.py helper to regenerate cast_list.json after config changes (S)

### Database Setup
- ✅ **Complete** - Done: 🔴 Design SQLite schema (storage/schema.sql) (M)
- ✅ **Complete** - Done: 🔴 Implement database connection manager (storage/db.py) (M)
- ✅ **Complete** - Done: 🟡 Create database migrations system (storage/migrations/) (L)
- ✅ **Complete** - Done: 🟡 Implement voice bank manager (storage/voice_bank_manager.py) (L)
- ✅ **Complete** - Done: 🟢 Create database backup utility (storage/backup.py) (M)
- ✅ **Complete** - Done: 🟢 Implement database integrity checks (S)

### File System Structure
- ✅ **Complete** - Done: 🔴 Implement paths manager (storage/paths.py) (M)
- ✅ **Complete** - Done: 🟡 Create data directory initialization script (scripts/setup/init_data_dirs.py) (S)
- ✅ **Complete** - Done: 🟡 Implement file naming conventions (storage/naming.py) (S)
- ✅ **Complete** - Done: 🟢 Setup automatic backup directories (S)

### Scripts & Utilities
- ✅ **Complete** - Done: 🔴 Create setup_ffmpeg_mac.sh (scripts/setup/setup_ffmpeg_mac.sh) (S)
- ✅ **Complete** - Done: 🔴 Create download_models.py (scripts/setup/download_models.py) (M)
- ✅ **Complete** - Done: 🟡 Create verify_dependencies.py (scripts/setup/verify_dependencies.py) (S)
- ✅ **Complete** - Done: 🟢 Create cleanup_cache.py (scripts/maintenance/cleanup_cache.py) (S)

---

## Core Pipeline - Stage 1: Audio Extraction

### FFmpeg Integration
- ✅ **Complete** - Done: 🔴 Implement FFmpeg wrapper (utils/ffmpeg.py) (M)
- ✅ **Complete** - Done: 🔴 Create audio extraction pipeline (pipelines/extract_audio.py) (M)
  - ✅ **Complete** - Video → WAV conversion (16kHz mono)
  - ✅ **Complete** - Audio normalization (-20dB LUFS)
  - ✅ **Complete** - Format validation
  - ✅ **Complete** - Error handling
- ✅ **Complete** - Done: 🟡 Implement audio quality checks (S)
- ✅ **Complete** - Done: 🟡 Add progress reporting (S)
- ✅ **Complete** - Done: 🟢 Support multiple video formats (MP4, MKV, AVI, MOV) (S)
- ✅ **Complete** - Done: 🟢 Implement audio file validation (S)

### Audio I/O Utilities
- ✅ **Complete** - Done: 🔴 Create audio_io.py utility module (utils/audio_io.py) (M)
  - ✅ **Complete** - Load audio files (WAV, MP3)
  - ✅ **Complete** - Resample audio
  - ✅ **Complete** - Convert formats
  - ✅ **Complete** - Extract audio segments
- ✅ **Complete** - Done: 🟡 Implement audio visualization helpers (utils/audio_visualization.py) (S)

### Testing
- ✅ **Complete** - Done: 🟡 Unit tests for FFmpeg wrapper (tests/unit/utils/test_ffmpeg.py) (M)
- ✅ **Complete** - Done: 🟡 Integration test for full extraction pipeline (tests/integration/test_extract_audio_pipeline.py) (S)
- ✅ **Complete** - Done: 🟢 Edge case tests (corrupted files, unsupported formats) (tests/unit/pipelines/test_extract_audio.py) (M)

---

## Core Pipeline - Stage 2: Transcription (ASR)

### Whisper Local Implementation
- ✅ **Complete** - Done: 🔴 Implement local Whisper interface (pipelines/asr/whisper_local.py) (L)
  - ✅ **Complete** - Model loading and caching
  - ✅ **Complete** - Transcription with word-level timestamps
  - ✅ **Complete** - Confidence score extraction
  - ✅ **Complete** - Multi-language support
- ✅ **Complete** - Done: 🟡 Implement model download/verification (M)
- ✅ **Complete** - Done: 🟡 Add GPU/Neural Engine acceleration (M)
- ✅ **Complete** - Done: 🟢 Implement model size selection (large/medium/small) (S)

### Whisper API Implementation
- ✅ **Complete** - Done: 🔴 Implement OpenAI Whisper API client (pipelines/asr/whisper_api.py) (M)
  - ✅ **Complete** - API authentication
  - ✅ **Complete** - File upload and transcription
  - ✅ **Complete** - Response parsing
  - ✅ **Complete** - Error handling and retries
- ✅ **Complete** - Done: 🟡 Implement rate limiting handling (S)
- ✅ **Complete** - Done: 🟡 Configure API credentials in `.env` (XS)
- ✅ **Complete** - Done: 🟡 Add cost tracking (S)
- ✅ **Complete** - Done: 🟢 Implement fallback to local on API failure (M)

### Alternative ASR Providers (Optional)
- [ ] 🔵 Implement ElevenLabs Scribe API (pipelines/asr/scribe_api.py) (M)
- [ ] 🔵 Implement AssemblyAI integration (providers/assemblyai_api.py) (M)

### ASR Output Processing
- [ ] 🔴 Standardize ASR output format (transcript_raw.json) (S)
- [ ] 🟡 Implement confidence filtering (S)
- [ ] 🟡 Handle multi-language detection (S)

### Testing
- ✅ **Complete** - Done: 🟡 Unit tests for Whisper local (tests/unit/pipelines/asr/test_whisper_local.py) (M)
- ✅ **Complete** - Done: 🟡 Unit tests for Whisper API (tests/unit/pipelines/asr/test_whisper_api.py) (M)
- ✅ **Complete** - Done: 🟡 Mock tests for API client (tests/unit/pipelines/asr/test_whisper_api.py) (S)
- [ ] 🟢 Accuracy benchmarks on sample audio (M)
- [ ] 🟢 Performance benchmarks (speed, memory) (M)

---

## Core Pipeline - Stage 3: Diarization

### Pyannote Integration
- ✅ **Complete** - Done: 🟡 Provision HuggingFace/Pyannote tokens in `.env` (XS)
- ✅ **Complete** - Done: 🔴 Implement Pyannote pipeline (pipelines/diarization/pyannote_pipeline.py) (L)
  - ✅ Pipeline initialization
  - ✅ Speaker segmentation
  - ✅ Voice activity detection
  - ✅ Overlapping speech handling
- [ ] 🟡 Implement HuggingFace authentication (S)
- [ ] 🟡 Add GPU acceleration support (S)
- [ ] 🟡 Configure speaker count parameters (min/max speakers) (S)

### Diarization Output Processing
- [ ] 🔴 Standardize diarization output format (diarization.json) (S)
- [ ] 🟡 Implement speaker cluster labeling (SPEAKER_00, SPEAKER_01, etc.) (S)
- [ ] 🟡 Handle overlapping speech segments (M)

### Alignment with ASR
- ✅ **Complete** - Done: 🔴 Implement ASR + Diarization alignment (pipelines/alignment/align_asr_diar.py) (L)
  - ✅ **Complete** - Time-based segment matching
  - ✅ **Complete** - Word-level speaker assignment
  - ✅ **Complete** - Conflict resolution (overlapping speakers)
  - ✅ **Complete** - Confidence scoring
- ✅ **Complete** - Follow-up: Ensure transcript exports leverage alignment metadata (export_text/export_srt/export_json).
- ✅ **Complete** - Follow-up: Capture transcription cost metrics once API fallback is wired (align with Stage 2 cost tracking).

### Testing
- [ ] 🟡 Unit tests for diarization (tests/unit/pipelines/test_diarization.py) (M)
- [ ] 🟡 Test overlapping speech handling (M)
- [ ] 🟡 Test alignment algorithm (M)
- [ ] 🟢 DER (Diarization Error Rate) benchmarks (M)
- [ ] 🟢 Edge cases: single speaker, many speakers, no speech (M)

---

## Core Pipeline - Stage 3b: Bleep Detection

### DSP Feature Extraction
- [ ] 🔴 Implement DSP features module (pipelines/bleep_detection/dsp_features.py) (L)
  - [ ] STFT (Short-Time Fourier Transform)
  - [ ] RMS energy calculation
  - [ ] Spectral centroid extraction
  - [ ] Spectral bandwidth calculation
  - [ ] Crest factor computation
  - [ ] Spectral flatness measurement

### Bleep Detection Algorithm
- [ ] 🔴 Implement bleep detector (pipelines/bleep_detection/detector.py) (XL)
  - [ ] Tone detection (narrowband, 0.8-2.5 kHz)
  - [ ] Mute detection (silence drops during speech)
  - [ ] Noise detection (broadband bursts)
  - [ ] Event merging (gap < 120ms)
  - [ ] Duration filtering (min 80ms)
  - [ ] Confidence scoring

### SFX Profile Learning
- [ ] 🟡 Implement SFX profile system (pipelines/bleep_detection/sfx_profiles.py) (L)
  - [ ] Learn recurring bleep signatures
  - [ ] Store in database (sfx_profiles table)
  - [ ] Match against learned profiles
  - [ ] Update profiles over time

### Bleep Attribution
- [ ] 🔴 Align bleep events with speaker segments (M)
- [ ] 🟡 Assign bleeps to speakers based on overlap (S)
- [ ] 🟡 Handle bleeps at speaker boundaries (S)

### Word Suggestion (Optional)
- [ ] 🟢 Implement context-based word suggestion (pipelines/bleep_detection/suggest_word.py) (L)
  - [ ] Analyze surrounding text
  - [ ] Use GPT for suggestions (optional)
  - [ ] Rank suggestions by likelihood

### Testing
- [ ] 🟡 Unit tests for DSP features (tests/unit/pipelines/test_bleep_detector.py) (M)
- [ ] 🟡 Test tone detection with synthetic bleeps (M)
- [ ] 🟡 Test mute detection (M)
- [ ] 🟡 Test noise detection (M)
- [ ] 🟢 Precision/recall benchmarks (M)
- [ ] 🟢 False positive testing (background music, effects) (M)

---

## Core Pipeline - Stage 4: Speaker Identification

### Voice Embedding Generation
- [ ] 🔴 Implement Resemblyzer backend (pipelines/embeddings/resemblyzer_backend.py) (M)
  - [ ] Load encoder model
  - [ ] Generate 256-d embeddings
  - [ ] Handle audio preprocessing
- [ ] 🟡 Implement Pyannote embeddings backend (pipelines/embeddings/pyannote_embeddings.py) (M)
- [ ] 🟡 Implement embedding caching (S)

### Voice Bank Matching
- [ ] 🔴 Implement speaker matcher (pipelines/speaker_id/matcher.py) (L)
  - [ ] Cosine similarity calculation
  - [ ] Confidence thresholding
  - [ ] Multi-embedding matching (avg/max pooling)
  - [ ] Conflict resolution

### Confidence Calibration
- [ ] 🟡 Implement thresholding system (pipelines/speaker_id/thresholding.py) (M)
  - [ ] High confidence: ≥0.85
  - [ ] Medium confidence: 0.70-0.85
  - [ ] Low confidence: <0.70
- [ ] 🟡 Implement adaptive threshold adjustment (M)

### Voice Bank CRUD
- [ ] 🔴 Implement voice bank operations (pipelines/speaker_id/voice_bank.py) (L)
  - [ ] Add speaker profile
  - [ ] Add voice embedding
  - [ ] Update speaker info
  - [ ] Delete speaker
  - [ ] Merge speakers
  - [ ] Export/import voice bank

### Testing
- [ ] 🟡 Unit tests for embeddings (tests/unit/pipelines/test_embeddings.py) (M)
- [ ] 🟡 Unit tests for matcher (tests/unit/pipelines/test_speaker_id.py) (M)
- [ ] 🟡 Test voice bank operations (M)
- [ ] 🟢 Accuracy benchmarks (M)
- [ ] 🟢 Test similar voices (twins, family) (M)
- [ ] 🟢 Test voice effects (phone, distortion) (M)

---

## Voice Bank System

### Database Operations
- [ ] 🔴 Implement speaker profile CRUD (storage/db.py) (M)
- [ ] 🔴 Implement embedding storage/retrieval (M)
- [ ] 🟡 Implement processing history tracking (S)
- [ ] 🟡 Implement episode-speaker relationships (S)
- [ ] 🟡 Implement bleep event storage (S)

### File System Management
- [ ] 🔴 Implement embedding file storage (.npy files) (M)
- [ ] 🟡 Implement audio sample storage (.wav clips) (S)
- [ ] 🟡 Implement orphan cleanup (remove unused files) (M)

### Voice Bank Management Scripts
- [ ] 🟡 Create export_voice_bank.py (scripts/voice_bank/export_voice_bank.py) (M)
- [ ] 🟡 Create import_voice_bank.py (scripts/voice_bank/import_voice_bank.py) (M)
- [ ] 🟡 Create merge_speakers.py (scripts/voice_bank/merge_speakers.py) (L)
- [ ] 🟡 Create cleanup_voice_bank.py (scripts/voice_bank/cleanup_voice_bank.py) (M)
- [ ] 🟢 Create rebuild_voice_bank.py (scripts/voice_bank/rebuild_voice_bank.py) (M)

### Backup & Restore
- [ ] 🟡 Implement automated backup (storage/backup.py) (M)
- [ ] 🟡 Implement restore from backup (M)
- [ ] 🟢 Setup scheduled backups (daily cron) (S)

### Testing
- [ ] 🟡 Unit tests for database operations (tests/unit/storage/test_db.py) (M)
- [ ] 🟡 Unit tests for voice bank manager (tests/unit/storage/test_voice_bank_manager.py) (M)
- [ ] 🟡 Integration tests for backup/restore (M)

---

## Transcript Generation & Export

### Transcript Builder
- ✅ **Complete** - Done: 🔴 Implement transcript builder (pipelines/transcript/builder.py) (L)
  - ✅ Merge ASR + diarization + bleeps
  - ✅ Insert speaker labels
  - ✅ Insert [BLEEP] tokens
  - ✅ Format timestamps
  - ✅ Handle overlapping speech

### Export Formats
- [ ] 🔴 Implement plain text export (pipelines/transcript/export_text.py) (M)
  - [ ] Speaker-labeled format
  - [ ] Timestamp formatting
  - [ ] Episode metadata header
- [ ] 🔴 Implement SRT export (pipelines/transcript/export_srt.py) (M)
  - [ ] SRT subtitle format
  - [ ] Speaker name in brackets
  - [ ] Proper timecode format
- [ ] 🔴 Implement JSON export (pipelines/transcript/export_json.py) (M)
  - [ ] Full structured data
  - [ ] All metadata included
  - [ ] Segments, speakers, bleeps

### Bleep Export
- [ ] 🔴 Implement bleeps CSV export (M)
  - [ ] Format: WORD, PERSON, TIMESTAMP, SENTENCE
  - [ ] Sortable by person/time
- [ ] 🟡 Implement bleeps JSON export (S)
  - [ ] Detailed event data
  - [ ] Confidence scores
  - [ ] Audio snippet paths

### Testing
- ✅ **Complete** - Done: 🟡 Unit tests for transcript builder (tests/unit/pipelines/test_transcript_builder.py) (M)
- [ ] 🟡 Test all export formats (M)
- [ ] 🟢 Validate SRT format compliance (S)
- [ ] 🟢 Test with edge cases (no dialogue, all bleeps, etc.) (M)

---

## Analytics & Reporting

### Speaking Time Calculator
- [ ] 🔴 Implement speaking time analytics (pipelines/analytics/speaking_time.py) (M)
  - [ ] Calculate duration per speaker
  - [ ] Calculate word count per speaker
  - [ ] Calculate segment count per speaker
  - [ ] Calculate percentage of total dialogue

### Bleep Statistics
- [ ] 🟡 Implement bleep statistics (pipelines/analytics/bleep_stats.py) (M)
  - [ ] Count by type (tone/mute/noise)
  - [ ] Count by person
  - [ ] Rate per minute
  - [ ] Temporal distribution

### Report Generation
- [ ] 🟡 Implement analytics report builder (pipelines/analytics/reports.py) (M)
  - [ ] JSON format (analytics.json)
  - [ ] Summary statistics
  - [ ] Per-speaker breakdown
  - [ ] Bleep analysis

### Future Analytics (v1.5+)
- [ ] 🔵 Cross-episode analytics (pipelines/analytics/cross_episode.py) (L)
- [ ] 🔵 Speaker interaction patterns (pipelines/analytics/speaker_patterns.py) (L)
- [ ] 🔵 CSV/Excel export (pipelines/analytics/export_analytics.py) (M)

### Testing
- [ ] 🟡 Unit tests for analytics (tests/unit/pipelines/test_analytics.py) (M)
- [ ] 🟢 Validate calculations with known data (M)

---

## Checkpoint & Resume System

### Checkpoint Manager
- [ ] 🔴 Implement checkpoint manager (pipelines/checkpoint/manager.py) (L)
  - [ ] Create checkpoint flags
  - [ ] Verify checkpoint integrity
  - [ ] Clear checkpoints
  - [ ] List checkpoints

### Resume Logic
- [ ] 🔴 Implement resume functionality (pipelines/checkpoint/resume.py) (L)
  - [ ] Detect last completed stage
  - [ ] Resume from checkpoint
  - [ ] Skip completed stages
  - [ ] Validate intermediate outputs

### State Tracking
- [ ] 🟡 Implement processing state tracker (pipelines/checkpoint/state.py) (M)
  - [ ] Track current stage
  - [ ] Track progress percentage
  - [ ] Track errors and retries

### Testing
- [ ] 🟡 Unit tests for checkpoint system (tests/unit/pipelines/test_checkpoint.py) (M)
- [ ] 🟡 Integration tests for resume (tests/integration/test_resume_pipeline.py) (L)
- [ ] 🟢 Test resume after each stage failure (L)

---

## User Interface (Streamlit)

### Main Application
- 🚧 In Progress - 🔴 Create Streamlit app entry point (ui/streamlit_app/app.py) (M)
  - ✅ Alignment preview tab renders via `render_alignment_view`
  - ✅ Episode selection + session state wired to `outputs/`
  - ✅ Navigation sidebar
  - ✅ Page routing across core review views
  - ✅ Session state management for selections

### Page 1: Process Episode
- [ ] 🔴 Create episode processing page (ui/streamlit_app/pages/1_Process_Episode.py) (L)
  - [ ] File upload widget
  - [ ] Episode configuration form
  - [ ] Start processing button
  - [ ] Progress display
  - [ ] Resume capability

### Page 2: Review Speakers
- [ ] 🔴 Create speaker review page (ui/streamlit_app/pages/2_Review_Speakers.py) (L)
  - [ ] Unknown speakers table
  - [ ] Audio playback for each segment
  - [ ] Speaker labeling form
  - [ ] Confidence badge display
  - [ ] Bulk labeling options

### Page 3: Voice Bank Management
- [ ] 🟡 Create voice bank page (ui/streamlit_app/pages/3_Voice_Bank.py) (L)
  - [ ] Speaker list with details
  - [ ] Add/edit/delete speakers
  - [ ] Merge speakers UI
  - [ ] View embeddings count
  - [ ] Export/import functionality

### Page 4: Bleep Review
- [ ] 🔴 Create bleep review page (ui/streamlit_app/pages/4_Bleep_Review.py) (L)
  - [ ] Bleeps table (sortable/filterable)
  - [ ] Audio playback for each bleep
  - [ ] Word label editing
  - [ ] Sentence context display
  - [ ] Bulk operations

### Page 5: Analytics
- [ ] 🟡 Create analytics dashboard (ui/streamlit_app/pages/5_Analytics.py) (L)
  - [ ] Speaking time charts
  - [ ] Bleep statistics charts
  - [ ] Episode comparison (future)
  - [ ] Export analytics data

### Page 6: Settings
- [ ] 🟢 Create settings page (ui/streamlit_app/pages/6_Settings.py) (M)
  - [ ] Configuration editor
  - [ ] API key management
  - [ ] Path configuration
  - [ ] Backup settings

### Reusable Components
- [ ] 🔴 Create audio player component (ui/streamlit_app/components/audio_player.py) (M)
- [ ] 🟡 Create speaker table component (ui/streamlit_app/components/speaker_table.py) (M)
- [ ] 🟡 Create bleep table component (ui/streamlit_app/components/bleep_table.py) (M)
- [ ] 🟢 Create waveform visualization (ui/streamlit_app/components/waveform_viz.py) (L)
- [ ] 🟢 Create confidence badge component (ui/streamlit_app/components/confidence_badge.py) (S)
- [ ] 🟢 Create merge speakers UI (ui/streamlit_app/components/merge_speakers_ui.py) (M)

### Testing
- [ ] 🟢 Manual UI testing checklist (M)
- [ ] 🟢 Create UI test fixtures (S)

---

## CLI Interface

### Main CLI Entry Point
- [ ] 🔴 Implement main CLI (src/show_scribe/cli.py) (L)
  - [ ] Argument parsing (argparse/click)
  - [ ] Subcommand routing
  - [ ] Error handling
  - [ ] Help text

### CLI Commands
- [ ] 🔴 `show-scribe process <video>` - Process episode (M)
- [ ] 🔴 `show-scribe resume <episode_id>` - Resume processing (S)
- [ ] 🟡 `show-scribe ui` - Launch Streamlit UI (S)
- [ ] 🟡 `show-scribe voice-bank export <output>` - Export voice bank (S)
- [ ] 🟡 `show-scribe voice-bank import <input>` - Import voice bank (S)
- [ ] 🟡 `show-scribe voice-bank verify` - Check integrity (S)
- [ ] 🟡 `show-scribe backup` - Create backup (S)
- [ ] 🟡 `show-scribe restore <backup>` - Restore from backup (S)
- [ ] 🟡 `show-scribe download-models` - Download AI models (S)
- [ ] 🟢 `show-scribe metrics <episode_id>` - Show metrics (S)
- [ ] 🟢 `show-scribe --version` - Display version (XS)
- [ ] 🟢 `show-scribe --help` - Show help (XS)

### Progress Reporting
- [ ] 🔴 Implement progress bars (tqdm) (S)
- [ ] 🟡 Implement stage-by-stage progress (S)
- [ ] 🟡 Implement time estimates (S)

### Testing
- [ ] 🟡 Unit tests for CLI (tests/unit/test_cli.py) (M)
- [ ] 🟢 Integration tests for each command (L)

---

## Testing & Quality Assurance

### Unit Tests
- [ ] 🟡 Audio extraction tests (tests/unit/pipelines/test_extract_audio.py) (M)
- [ ] 🟡 ASR tests (tests/unit/pipelines/test_asr.py) (M)
- [ ] 🟡 Diarization tests (tests/unit/pipelines/test_diarization.py) (M)
- [ ] 🟡 Bleep detection tests (tests/unit/pipelines/test_bleep_detector.py) (M)
- [ ] 🟡 Speaker ID tests (tests/unit/pipelines/test_speaker_id.py) (M)
- [ ] 🟡 Embeddings tests (tests/unit/pipelines/test_embeddings.py) (M)
- [ ] 🟡 Transcript tests (tests/unit/pipelines/test_transcript.py) (M)
- [ ] 🟡 Analytics tests (tests/unit/pipelines/test_analytics.py) (M)
- [ ] 🟡 Database tests (tests/unit/storage/test_db.py) (M)
- [ ] 🟡 Voice bank manager tests (tests/unit/storage/test_voice_bank_manager.py) (M)
- [ ] 🟡 Backup tests (tests/unit/storage/test_backup.py) (M)

### Integration Tests
- [ ] 🟡 End-to-end pipeline test (tests/integration/test_pipeline_e2e.py) (L)
- [ ] 🟡 Resume pipeline test (tests/integration/test_resume_pipeline.py) (L)
- [ ] 🟡 Voice bank growth test (tests/integration/test_voice_bank_growth.py) (M)

### Edge Case Tests
- [ ] 🟢 Background noise test (tests/edge_cases/test_background_noise.py) (M)
- [ ] 🟢 Overlapping speech test (tests/edge_cases/test_overlapping_speech.py) (M)
- [ ] 🟢 Similar voices test (tests/edge_cases/test_similar_voices.py) (M)
- [ ] 🟢 Voice effects test (tests/edge_cases/test_voice_effects.py) (M)
- [ ] 🟢 Single speaker episode test (M)
- [ ] 🟢 No dialogue episode test (M)
- [ ] 🟢 Long episode (>2 hours) test (M)
- [ ] 🟢 Multi-language test (M)

### Performance Tests
- [ ] 🟢 Benchmark ASR speed (M)
- [ ] 🟢 Benchmark diarization speed (M)
- [ ] 🟢 Benchmark speaker ID speed (M)
- [ ] 🟢 Memory usage profiling (M)
- [ ] 🟢 Disk I/O profiling (M)

### Test Coverage
- [ ] 🟡 Setup pytest-cov (S)
- [ ] 🟡 Achieve >80% code coverage target (XL)
- [ ] 🟢 Generate HTML coverage reports (S)

### CI/CD
- [ ] 🟡 Setup GitHub Actions workflow (.github/workflows/ci.yml) (M)
  - [ ] Run linting (Ruff + Black)
  - [ ] Run type checking (mypy)
  - [ ] Run unit tests
  - [ ] Run integration tests
  - [ ] Generate coverage report
- [ ] 🟢 Setup macOS runner (S)
- [ ] 🟢 Cache dependencies for faster builds (S)

---

## Documentation

### User Documentation
- [ ] 🟡 Complete README.md (M) ✅ DONE
- [ ] 🟡 Create installation guide (docs/installation.md) (M)
- [ ] 🟡 Create user guide (docs/user_guide.md) (L)
- [ ] 🟡 Create configuration guide (docs/configuration.md) (M)
- [ ] 🟢 Create troubleshooting guide (docs/troubleshooting.md) (M)
- [ ] 🟢 Create FAQ (docs/faq.md) (M)

### Technical Documentation
- [ ] 🟡 Complete PRD.md (L) ✅ DONE
- [ ] 🟡 Complete TECH_SPEC.md (L) ✅ DONE
- [ ] 🟡 Complete SOLUTION_ARCHITECTURE.md (XL) ✅ DONE
- [ ] 🟡 Complete DIRECTORY_STRUCTURE.md (M) ✅ DONE
- [ ] 🟡 Create API documentation (docs/api.md) (L)
- [ ] 🟢 Create database schema documentation (docs/database_schema.md) (M)
- [ ] 🟢 Create performance tuning guide (docs/performance_tuning.md) (M)

### Example Documentation
- [ ] 🟢 Create basic workflow example (docs/examples/basic_workflow.md) (M)
- [ ] 🟢 Create batch processing example (docs/examples/batch_processing.md) (M)
- [ ] 🟢 Create voice bank management example (docs/examples/voice_bank_management.md) (M)

### Code Documentation
- [ ] 🟡 Add docstrings to all public functions (XL)
- [ ] 🟡 Add type hints to all functions (XL)
- [ ] 🟢 Generate API documentation with Sphinx (M)

---

## Deployment & Distribution

### Package Configuration
- [ ] 🔴 Complete pyproject.toml (M)
  - [ ] Project metadata
  - [ ] Dependencies
  - [ ] Optional dependencies (dev, test)
  - [ ] Entry points (CLI commands)
  - [ ] Build system configuration

### Distribution
- [ ] 🟡 Create PyPI package (M)
- [ ] 🟡 Test pip install locally (S)
- [ ] 🟡 Publish to PyPI (test.pypi.org first) (M)
- [ ] 🟢 Create Homebrew formula (L)
- [ ] 🟢 Setup automated releases (GitHub Actions) (M)

### macOS Application (Future)
- [ ] 🔵 Create .app bundle with py2app (XL)
- [ ] 🔵 Code signing (M)
- [ ] 🔵 Notarization for Gatekeeper (M)
- [ ] 🔵 Create DMG installer (M)

### Installation Testing
- [ ] 🟡 Test on clean macOS Monterey (M)
- [ ] 🟡 Test on macOS Ventura (M)
- [ ] 🟡 Test on macOS Sonoma (M)
- [ ] 🟡 Test on Apple Silicon (M1/M2/M3) (M)
- [ ] 🟡 Test on Intel Mac (M)

### Release Process
- [ ] 🟢 Create release checklist (S)
- [ ] 🟢 Setup semantic versioning (S)
- [ ] 🟢 Create CHANGELOG.md format (S)
- [ ] 🟢 Automate version bumping (S)

---

## Future Phases (v1.5 - v4.0)

### Phase 2: Automation & Scale (v1.5) - Q1 2026
- [ ] 🔵 Batch processing (multiple episodes in parallel) (L)
- [ ] 🔵 Improved confidence calibration (L)
- [ ] 🔵 Voice bank management UI enhancements (M)
- [ ] 🔵 Automated folder monitoring (L)
- [ ] 🔵 Cross-episode analytics dashboard (L)
- [ ] 🔵 Enhanced bleep word suggestion (context-aware) (L)

### Phase 3: Collaboration & Cloud (v2.0) - Q2 2026
- [ ] 🔵 User authentication and roles (XL)
- [ ] 🔵 Cloud storage integration (S3, Google Drive) (L)
- [ ] 🔵 Shared voice banks across team (L)
- [ ] 🔵 Real-time collaborative review (XL)
- [ ] 🔵 REST API for integrations (XL)
- [ ] 🔵 Webhook notifications (M)
- [ ] 🔵 Migrate to PostgreSQL (L)

### Phase 4: Intelligence & Integration (v3.0) - Q3 2026
- [ ] 🔵 Emotion/sentiment detection (XL)
- [ ] 🔵 Topic modeling and summarization (L)
- [ ] 🔵 Video editor plugins (Premiere Pro, Final Cut) (XL)
- [ ] 🔵 Real-time streaming transcription (XL)
- [ ] 🔵 Multi-language translation (L)
- [ ] 🔵 ML-enhanced bleep detection (L)

### Phase 5: Enterprise (v4.0) - Q4 2026+
- [ ] 🔵 Role-based access control (RBAC) (L)
- [ ] 🔵 Audit trails and compliance reporting (L)
- [ ] 🔵 Voice biometric encryption (L)
- [ ] 🔵 Multi-tenancy support (XL)
- [ ] 🔵 SLA-backed processing guarantees (XL)
- [ ] 🔵 BI integration (Tableau, PowerBI) (L)

---

## Project Milestones

### Milestone 1: Foundation (Week 1-2)
**Target:** Complete project setup and core infrastructure

**Critical Path:**
- [x] Repository setup
- [ ] Development environment
- [ ] Configuration system
- [ ] Database schema
- [ ] File system structure

### Milestone 2: Core Pipeline (Week 3-6)
**Target:** Implement end-to-end processing pipeline

**Critical Path:**
- [ ] Audio extraction (Stage 1)
- [ ] Transcription (Stage 2)
- [ ] Diarization (Stage 3)
- [ ] Bleep detection (Stage 3b)
- [ ] Speaker identification (Stage 4)
- [ ] Checkpoint system

### Milestone 3: Voice Bank & Review (Week 7-8)
**Target:** Complete voice bank system and review UI

**Critical Path:**
- [ ] Voice bank CRUD operations
- [ ] Streamlit UI (pages 1-4)
- [ ] Manual review workflow
- [ ] Backup/restore functionality

### Milestone 4: Export & Analytics (Week 9)
**Target:** Implement transcript generation and analytics

**Critical Path:**
- [ ] Transcript builder
- [ ] Export formats (TXT, SRT, JSON, CSV)
- [ ] Analytics calculator
- [ ] CLI interface

### Milestone 5: Testing & Polish (Week 10-11)
**Target:** Comprehensive testing and bug fixes

**Critical Path:**
- [ ] Unit test suite (>80% coverage)
- [ ] Integration tests
- [ ] Edge case testing
- [ ] Performance optimization
- [ ] Bug fixes

### Milestone 6: Documentation & Release (Week 12)
**Target:** Complete documentation and v1.0 release

**Critical Path:**
- [ ] User documentation
- [ ] API documentation
- [ ] PyPI package
- [ ] Release announcement
- [ ] v1.0 launch! 🎉

---

## Daily Standup Tracking

### Today's Focus (Date: _______)
- [ ] Task 1
- [ ] Task 2
- [ ] Task 3

### Blockers
- None / List blockers here

### Yesterday's Accomplishments
- Completed: ___
- Completed: ___

### Notes
-

---

## Quick Reference

### High-Priority Tasks (Start Here)
1. 🔴 Repository setup
2. 🔴 Database schema implementation
3. 🔴 Audio extraction pipeline
4. 🔴 Whisper integration (local + API)
5. 🔴 Pyannote diarization
6. 🔴 Bleep detection DSP
7. 🔴 Speaker identification
8. 🔴 Checkpoint system
9. 🔴 Streamlit review UI
10. 🔴 Transcript generation

### Dependencies Graph
```
Repository Setup
    ↓
Configuration System
    ↓
Database Schema
    ↓
┌───────────────┴────────────────┐
│                                │
Audio Extraction → ASR → Diarization → Bleep Detection
                    ↓                       ↓
            Alignment ←──────────────────────┘
                    ↓
            Voice Embeddings
                    ↓
            Speaker Identification
                    ↓
            Manual Review (UI)
                    ↓
            Transcript Builder
                    ↓
            Analytics & Export
```

### Estimated Timeline
- **Phase 1 (MVP v1.0):** 12 weeks (3 months)
- **Total Tasks:** ~200+
- **Team Size:** 1-2 developers
- **Launch Target:** January 2026

---

## Notes & Ideas

### Implementation Notes
- Start with local Whisper (easier testing) before API integration
- Use sample audio/video files for development (5-10 min clips)
- Build UI incrementally (one page at a time)
- Focus on checkpoint system early - saves debugging time

### Future Considerations
- Consider Electron wrapper for better distribution
- Explore ONNX for faster inference
- Consider WebAssembly for browser-based processing
- Look into Opus codec for better audio compression

---

**Last Updated:** October 16, 2025
**Status:** Ready for Implementation
**Next Review:** Weekly during MVP development

---

*This is a living document. Update task status and add notes as development progresses.*
