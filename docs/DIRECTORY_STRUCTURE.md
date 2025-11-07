# Show-Scribe Repository Structure

**Project:** Show-Scribe Audio Analyzer
**Date:** October 16, 2025
**Version:** 1.0

---

## **Repository Layout**

```
show-scribe/
├── README.md
├── LICENSE
├── pyproject.toml                 # Python packaging (ruff/black/mypy/pytest in tool deps)
├── Makefile                       # setup, dev, lint, test, run, download-models, smoke
├── .gitignore
├── .env.example
├── configs/
│   ├── dev.yaml                   # local paths, thresholds, provider toggles
│   └── prod.yaml
├── docs/
│   ├── PRD.md
│   ├── addendum_bleeps.md
│   ├── architecture.md
│   ├── api.md
│   └── user_guide.md
├── scripts/
│   ├── setup/
│   │   ├── setup_ffmpeg_mac.sh    # brew install ffmpeg etc.
│   │   ├── download_models.py     # whisper/pyannote/resemblyzer weights
│   │   └── verify_dependencies.py # sanity checks for system/python deps
│   ├── maintenance/
│   │   └── cleanup_cache.py       # purge cache/temp directories
│   │   └── sync_cast_configs.py   # regenerate cast_list.json from show configs
│   ├── run_pipeline.py            # CLI driver: process one episode
│   ├── rebuild_voice_bank.py
│   └── export_episode_bundle.py
├── src/
│   └── show_scribe/
│       ├── __init__.py
│       ├── cli.py                 # `show-scribe process <video.mp4>` etc.
│       ├── config/
│       │   ├── __init__.py
│       │   ├── load.py
│       │   └── schema.json
│       ├── pipelines/
│       │   ├── ingest.py          # validate inputs, episode id generation
│       │   ├── extract_audio.py   # ffmpeg wrapper: MP4 → WAV/MP3
│       │   ├── asr/
│       │   │   ├── __init__.py
│       │   │   ├── whisper_local.py
│       │   │   ├── whisper_api.py
│       │   │   └── scribe_api.py  # alt: ElevenLabs/other
│       │   ├── diarization/
│       │   │   ├── __init__.py
│       │   │   └── pyannote_pipeline.py
│       │   ├── embeddings/
│       │   │   ├── __init__.py
│       │   │   ├── resemblyzer_backend.py
│       │   │   └── pyannote_embeddings.py
│       │   ├── speaker_id/
│       │   │   ├── __init__.py
│       │   │   ├── matcher.py     # cosine sim, thresholds, calibration
│       │   │   ├── thresholding.py
│       │   │   └── voice_bank.py  # CRUD for profiles/embeddings
│       │   ├── bleep_detection/
│       │   │   ├── __init__.py
│       │   │   ├── detector.py    # DSP+optional model; tone/mute/noise events
│       │   │   ├── dsp_features.py
│       │   │   ├── sfx_profiles.py # learnable BLEEP/SFX signatures
│       │   │   └── suggest_word.py # optional context-based suggestion
│       │   ├── alignment/
│       │   │   ├── __init__.py
│       │   │   └── align_asr_diar.py
│       │   ├── transcript/
│       │   │   ├── __init__.py
│       │   │   ├── builder.py     # speaker-labeled segments
│       │   │   ├── export_text.py
│       │   │   ├── export_srt.py
│       │   │   └── export_json.py
│       │   ├── analytics/
│       │   │   ├── __init__.py
│       │   │   ├── speaking_time.py
│       │   │   ├── bleep_stats.py
│       │   │   └── reports.py
│       │   └── review/
│       │       ├── __init__.py
│       │       ├── queue.py       # unknown speakers & bleep events
│       │       └── snippets.py    # audio clip generation
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── db.py              # SQLite; episodes, segments, bleeps, profiles
│       │   ├── schema.sql
│       │   └── paths.py           # standardized dirs/filenames
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── audio_io.py
│       │   ├── ffmpeg.py
│       │   ├── timecode.py
│       │   ├── logging.py
│       │   ├── name_correction.py # ⭐ NEW - Cast name auto-correction
│       │   └── file_ops.py
│       └── providers/             # optional integrations (feature-flagged)
│           ├── __init__.py
│           ├── openai_gpt.py      # summaries/QA (optional)
│           ├── elevenlabs_api.py  # Scribe ASR alt
│           └── assemblyai_api.py  # diarized ASR alt
├── ui/
│   ├── streamlit_app/             # lightweight review UI
│   │   ├── app.py                 # review unknown speakers, bleeps table
│   │   └── components/            # audio player, tables, filters
│   │       ├── audio_player.py
│   │       ├── speaker_table.py
│   │       ├── bleep_table.py
│   │       ├── name_selector.py   # ⭐ NEW - Validated name dropdown
│   │       └── waveform_viz.py
│   └── web/                       # (optional) Flask/React placeholder
├── data/
│   ├── episodes/                  # RAW INPUTS (gitignored)
│   ├── voice_bank/                # embeddings, profiles (gitignored)
│   ├── cache/                     # temp artifacts (gitignored)
│   └── samples/                   # tiny demo clips for tests/docs
├── outputs/
│   └── {episode_id}/              # transcript.txt/json, .srt, bleeps.csv/json, analytics.json
├── tests/
│   ├── unit/
│   │   ├── test_extract_audio.py
│   │   ├── test_asr.py
│   │   ├── test_diarization.py
│   │   ├── test_embeddings.py
│   │   ├── test_speaker_id.py
│   │   ├── test_bleep_detector.py
│   │   ├── test_transcript.py
│   │   └── test_analytics.py
│   ├── integration/
│   │   └── test_pipeline_e2e.py
│   └── fixtures/
│       ├── audio/
│       └── video/
├── .github/
│   └── workflows/
│       └── ci.yml                 # macOS runner: lint, typecheck, tests
├── .pre-commit-config.yaml
├── ruff.toml
└── mypy.ini
```

---

## **Recommended Enhancements**

Based on PRD requirements, consider these additions:

### **1. Data Structure Alignment**

```
data/
├── voice_bank/
│   ├── voice_bank.db              # SQLite database (ADD)
│   ├── embeddings/                # .npy files (ADD)
│   └── audio_samples/             # Reference clips (ADD)
├── shows/                         # CHANGE from episodes/
│   └── [ShowName]/
│       ├── show_config.json       # Per-show settings (ADD)
│       └── episodes/
│           └── S01E01/
│               ├── metadata.json            # Includes preprocessing_report_path when available
│               ├── audio_extracted.wav
│               ├── audio_processed.wav      # Final ASR audio (if preprocessing enabled)
│               ├── processed_audio/         # Preprocessing artifacts
│               │   ├── audio_vocals.wav
│               │   ├── audio_enhanced_vocals.wav
│               │   ├── audio_enhanced_mix.wav
│               │   └── preprocessing_report.json
│               ├── transcript_raw.json
│               ├── diarization.json
│               ├── transcript_final.txt
│               ├── transcript_final.srt
│               ├── transcript_final.json
│               ├── analytics.json
│               ├── bleeps.csv             # Bleep table (ADD)
│               ├── bleeps.json            # Bleep details (ADD)
│               └── checkpoints/           # Resume capability (ADD)
│                   ├── stage_1_complete.flag
│                   ├── stage_2_complete.flag
│                   └── ...
├── config/                        # MOVE from root
│   └── app_settings.json          # Global settings
├── backups/                       # ADD
│   └── voice_bank_YYYYMMDD_HHMMSS.db
├── cache/                         # Keep for temp artifacts
└── samples/                       # Keep for demo clips
```

**Remove:** `outputs/` directory - consolidate under `data/shows/`

### **2. Storage Module Enhancements**

```
storage/
├── __init__.py
├── db.py
├── schema.sql
├── paths.py
├── migrations/                    # ADD - schema updates
│   ├── __init__.py
│   ├── 001_initial.sql
│   ├── 002_add_bleeps.sql
│   └── migrate.py
├── backup.py                      # ADD - voice bank backup/restore
└── voice_bank_manager.py          # ADD - import/export operations
```

### **3. Enhanced Scripts Organization**

```
scripts/
├── setup/
│   ├── setup/
│   │   ├── setup_ffmpeg_mac.sh
│   │   ├── download_models.py
│   │   └── verify_dependencies.py
│   ├── maintenance/
│   │   └── cleanup_cache.py
│   └── verify_dependencies.py
├── pipeline/
│   ├── run_pipeline.py
│   ├── batch_process.py           # ADD - multiple episodes
│   └── resume_failed.py           # ADD - resume from checkpoint
├── voice_bank/
│   ├── rebuild_voice_bank.py
│   ├── export_voice_bank.py       # ADD - per PRD Section 10.4
│   ├── import_voice_bank.py       # ADD
│   ├── merge_speakers.py          # ADD - per PRD Section 10.3
│   └── cleanup_voice_bank.py      # ADD - remove low-quality samples
├── maintenance/
│   ├── backup_database.py         # ADD
│   ├── restore_database.py        # ADD
│   └── cleanup_cache.py
└── utils/
    ├── export_episode_bundle.py
    └── validate_data.py           # ADD - check integrity
```

### **4. Checkpoint/Resume System**

```
pipelines/
├── checkpoint/                    # ADD
│   ├── __init__.py
│   ├── manager.py                 # Create/verify/clear checkpoints
│   ├── resume.py                  # Resume logic from specific stage
│   └── state.py                   # Processing state tracking
```

### **5. Voice Bank Management UI**

```
ui/
├── streamlit_app/
│   ├── app.py
│   ├── pages/                     # ADD - multi-page app
│   │   ├── 1_Process_Episode.py
│   │   ├── 2_Review_Speakers.py
│   │   ├── 3_Voice_Bank.py        # NEW - browse/edit/merge speakers
│   │   ├── 4_Bleep_Review.py      # NEW - review bleeps
│   │   ├── 5_Analytics.py
│   │   └── 6_Settings.py
│   └── components/
│       ├── audio_player.py
│       ├── speaker_table.py
│       ├── bleep_table.py         # ADD
│       ├── waveform_viz.py        # ADD
│       ├── confidence_badge.py    # ADD
│       └── merge_speakers_ui.py   # ADD
```

### **6. Enhanced Analytics**

```
pipelines/analytics/
├── __init__.py
├── speaking_time.py
├── bleep_stats.py                 # Already included
├── reports.py
├── cross_episode.py               # ADD - season/show aggregation
├── speaker_patterns.py            # ADD - interaction analysis
└── export_analytics.py            # ADD - CSV/Excel export
```

### **7. Better Test Organization**

```
tests/
├── unit/
│   ├── pipelines/                 # Reorganize to mirror src/
│   │   ├── test_extract_audio.py
│   │   ├── test_asr.py
│   │   ├── test_diarization.py
│   │   ├── test_bleep_detector.py
│   │   └── ...
│   ├── storage/
│   │   ├── test_db.py
│   │   ├── test_voice_bank_manager.py
│   │   └── test_backup.py
│   └── utils/
├── integration/
│   ├── test_pipeline_e2e.py
│   ├── test_resume_pipeline.py    # ADD
│   └── test_voice_bank_growth.py  # ADD
├── edge_cases/                    # ADD - per PRD Section 8
│   ├── test_background_noise.py
│   ├── test_overlapping_speech.py
│   ├── test_similar_voices.py
│   └── test_voice_effects.py
└── fixtures/
    ├── audio/
    ├── video/
    └── voice_bank/                # ADD - sample voice bank
```

### **8. Enhanced Documentation**

```
docs/
├── PRD.md
├── addendum_bleeps.md
├── architecture.md
├── api.md
├── user_guide.md
├── installation.md                # ADD - detailed setup
├── configuration.md               # ADD - all config options
├── troubleshooting.md             # ADD - common issues
├── performance_tuning.md          # ADD - optimization tips
├── database_schema.md             # ADD - document schema
└── examples/                      # ADD
    ├── basic_workflow.md
    ├── batch_processing.md
    └── voice_bank_management.md
```

### **9. Root-Level Files**

```
show-scribe/
├── CHANGELOG.md                   # ADD - version history
├── CONTRIBUTING.md                # ADD - contribution guidelines
├── .python-version                # ADD - specify Python version
└── requirements-dev.txt           # ADD - or keep in pyproject.toml
```

---

## **Key Directories Explained**

### **`src/show_scribe/`**
Core application code following Python package structure.

### **`src/show_scribe/pipelines/`**
Processing stages: audio extraction, ASR, diarization, bleep detection, speaker ID, etc.

### **`src/show_scribe/storage/`**
Database operations, schema, file path management, backups.

### **`src/show_scribe/utils/`**
Shared utilities: audio I/O, FFmpeg wrappers, logging, timecode handling.

### **`ui/`**
User interface components. Streamlit for quick prototyping, optional web app for production.

### **`data/`**
**Gitignored.** All episode files, voice bank data, and processing outputs.

### **`scripts/`**
Standalone scripts for setup, maintenance, and one-off operations.

### **`tests/`**
Unit, integration, and edge case tests. Mirrors `src/` structure.

### **`configs/`**
YAML configuration files for different environments (dev, prod).

### **`docs/`**
All project documentation including PRD, architecture, API docs, user guides.

---

## **Makefile Targets**

Suggested Makefile commands:

```makefile
setup:              # Install dependencies, download models
dev:                # Run in dev mode with hot reload
lint:               # Run ruff + black
typecheck:          # Run mypy
test:               # Run pytest
test-unit:          # Run unit tests only
test-integration:   # Run integration tests
run:                # Run pipeline on example episode
download-models:    # Download Whisper, Pyannote models
smoke:              # Quick smoke test
clean:              # Clean cache, temp files
backup-voice-bank:  # Backup voice bank database
```

---

## **Git Ignore**

Key entries for `.gitignore`:

```
# Data directories
data/episodes/
data/voice_bank/
data/cache/
outputs/

# Models
*.bin
*.pt
*.onnx
models/

# Python
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Environment
.env
.venv/
venv/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

---

## **Development Workflow**

1. **Setup:**
   ```bash
   make setup
   ```

2. **Run on sample episode:**
   ```bash
   make run
   ```

3. **Development cycle:**
   ```bash
   make dev      # Start with hot reload
   make lint     # Before committing
   make test     # Run tests
   ```

4. **Process episode via CLI:**
   ```bash
   show-scribe process path/to/episode.mp4
   ```

---

## **Next Steps**

1. ✅ Implement data directory restructure (align with PRD Section 2.1)
2. ✅ Add checkpoint system for resume capability
3. ✅ Implement voice bank backup/restore
4. ✅ Create voice bank management scripts
5. ✅ Add bleep detection pipeline
6. ✅ Build Streamlit UI for review
7. ✅ Write comprehensive tests
8. ✅ Document API and workflows

---

**Document Version:** 1.0
**Last Updated:** October 16, 2025
**Status:** Implementation Ready
