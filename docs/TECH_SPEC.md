# Show-Scribe Technical Specification

**Version:** 1.1 (Updated with 2025 Best Practices)
**Date:** October 16, 2025
**Status:** Implementation Ready
**Changes:** Tech stack upgraded based on comprehensive review

---

## Table of Contents

1. [Technology Stack Overview](#technology-stack-overview)
2. [Core Dependencies](#core-dependencies)
3. [Architecture](#architecture)
4. [Development Stack](#development-stack)
5. [Performance Considerations](#performance-considerations)
6. [Security & Privacy](#security--privacy)
7. [Alternative Technologies Considered](#alternative-technologies-considered)
8. [Integration Specifications](#integration-specifications)
9. [Deployment & Distribution](#deployment--distribution)

---

## Technology Stack Overview

### Platform Requirements

| Component | Specification | Notes |
|-----------|--------------|-------|
| **Operating System** | macOS 12.0+ (Monterey) | Apple Silicon optimized |
| **Python Version** | 3.11.x | Latest stable CPython |
| **Memory** | 8 GB min, 16 GB recommended | For local Whisper model |
| **Storage** | 10 GB min, 50 GB recommended | Models + voice bank data |
| **Processor** | Apple Silicon (M1+) or Intel i5+ | Neural Engine support |

---

## Core Dependencies

### Speech Processing Stack

#### 1. **Faster-Whisper** (Speech-to-Text) ⭐ PRIMARY

```python
# Version: 1.0.0+ (4-10x faster than openai-whisper)
pip install faster-whisper>=1.0.0

# Alternative: WhisperX (best timestamps)
pip install whisperx
```

**Purpose:** Automatic Speech Recognition (ASR)

**Features:**
- 99 languages supported
- Word-level timestamps (especially accurate with WhisperX)
- Confidence scores per segment
- Robust to background noise
- **4-10x faster** than original Whisper
- **4x less memory** usage
- GPU acceleration (CUDA/CoreML)

**Model Size:**
- `large-v3`: 2.9 GB (same as original)
- `medium`: 1.5 GB
- `small`: 461 MB

**Performance:**
- Word Error Rate: 3-5% on clear audio (same as OpenAI Whisper)
- Processing Speed: **0.4-10x real-time** (vs 0.1-2x for original)
- Memory Usage: **1-2 GB** (vs 4-6 GB for original)
- API alternative: OpenAI Whisper API ($0.006/minute) as fallback

**Why Faster-Whisper:**
- CTranslate2 backend (optimized C++ implementation)
- Same accuracy, much better performance
- Actively maintained (2024 updates)
- Drop-in replacement for openai-whisper

**Integration Points:**
- `src/show_scribe/pipelines/asr/faster_whisper.py` (PRIMARY)
- `src/show_scribe/pipelines/asr/whisperx.py` (alternative for best timestamps)
- `src/show_scribe/pipelines/asr/whisper_api.py` (cloud fallback)

---

#### 2. **Pyannote.audio** (Speaker Diarization)

```python
# Version: 3.1.1
pip install pyannote.audio==3.1.1

# Required for embeddings
pip install pyannote.core==5.0.0
```

**Purpose:** Speaker segmentation and clustering

**Features:**
- Voice Activity Detection (VAD)
- Overlapping speech detection
- Speaker change point detection
- Speaker embeddings (512-dimensional)

**Models:**
- `pyannote/speaker-diarization@2.1`: ~300 MB
- `pyannote/embedding`: ~17 MB

**Performance:**
- Diarization Error Rate (DER): 10-15%
- Processing Speed: 0.5-1x real-time
- Memory: ~2-4 GB during processing

**Configuration:**
```python
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1",
    use_auth_token="YOUR_HF_TOKEN"
)

diarization = pipeline(
    audio_file,
    min_speakers=2,
    max_speakers=10
)
```

**Integration Points:**
- `src/show_scribe/pipelines/diarization/pyannote_pipeline.py`
- `src/show_scribe/pipelines/embeddings/pyannote_embeddings.py`

---

#### 3. **Pyannote Embeddings** (Voice Embeddings) ⭐ PRIMARY

```python
# Already included with pyannote.audio
from pyannote.audio import Model, Inference

model = Model.from_pretrained("pyannote/embedding")
```

**Purpose:** Speaker voice fingerprinting

**Features:**
- 512-dimensional voice embeddings (vs 256-d for older models)
- Speaker verification and identification
- Fast inference (<100ms per clip)
- Pre-trained on VoxCeleb datasets
- Actively maintained (2024 updates)
- Integrated with Pyannote diarization

**Model Size:** ~17 MB

**Why Pyannote Embeddings:**
- Better accuracy than Resemblyzer (10-15% improvement)
- Same ecosystem as diarization pipeline
- Modern architecture (trained 2023-2024)
- Actively maintained
- Better performance on diverse voices

**Alternative:** SpeechBrain ECAPA-TDNN embeddings (if SOTA accuracy needed)

**Integration Points:**
- `src/show_scribe/pipelines/embeddings/pyannote_embeddings.py` (PRIMARY)
- `src/show_scribe/pipelines/embeddings/speechbrain_backend.py` (alternative)
- `src/show_scribe/pipelines/speaker_id/matcher.py`

---

#### 4. **FFmpeg** (Media Processing)

```bash
# Installation via Homebrew
brew install ffmpeg

# Version: 6.0+
```

**Purpose:** Audio extraction and conversion

**Key Operations:**
```bash
# Extract audio from video
ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 output.wav

# Normalize audio levels
ffmpeg -i input.wav -af loudnorm=I=-20:TP=-1.5 output_normalized.wav

# Extract audio segment
ffmpeg -i input.wav -ss 00:01:30 -t 00:00:10 segment.wav
```

**Python Integration:**
```python
# Version: 0.2.0
pip install ffmpeg-python==0.2.0
```

**Integration Points:**
- `src/show_scribe/pipelines/extract_audio.py`
- `src/show_scribe/utils/ffmpeg.py`

---

### Signal Processing Stack

#### 5. **torchaudio** (GPU-Accelerated Audio Processing) ⭐ NEW

```python
# Version: 2.1.0+
pip install torchaudio>=2.1.0
```

**Purpose:** High-performance audio processing with GPU acceleration

**Features:**
- GPU acceleration (Metal Performance Shaders on Apple Silicon)
- 5-10x faster than librosa for spectrograms
- Rich transform library (STFT, Mel-spectrogram, etc.)
- PyTorch ecosystem integration
- I/O operations for common audio formats

**Usage:**
```python
import torchaudio
import torch

# Load audio with GPU support
waveform, sample_rate = torchaudio.load(audio_path)
waveform = waveform.to("mps")  # Apple Silicon GPU

# Fast spectrogram computation
spectrogram_transform = torchaudio.transforms.Spectrogram(
    n_fft=2048,
    hop_length=512
)
spectrogram = spectrogram_transform(waveform)
```

**Benefits:**
- Dramatically faster audio processing
- Reduced CPU load
- Better integration with ML pipelines
- Native GPU support

**Integration Points:**
- `src/show_scribe/utils/audio_io.py`
- `src/show_scribe/pipelines/bleep_detection/dsp_features.py`

---

#### 6. **soundfile** (Fast Audio I/O) ⭐ NEW

```python
# Version: 0.12.1
pip install soundfile>=0.12.1
```

**Purpose:** Fastest audio file reading/writing in Python

**Features:**
- 2-3x faster than librosa.load()
- Memory efficient
- No resampling overhead
- libsndfile wrapper
- Supports WAV, FLAC, OGG, etc.

**Usage:**
```python
import soundfile as sf

# Fast audio loading
audio, sample_rate = sf.read(audio_path)

# Fast audio writing
sf.write(output_path, audio, sample_rate)
```

**Benefits:**
- Significantly faster I/O operations
- Lower memory footprint
- Simple API

**Integration Points:**
- `src/show_scribe/utils/audio_io.py`
- `src/show_scribe/pipelines/extract_audio.py`

---

#### 7. **NumPy** (Numerical Computing)

```python
# Version: 1.24.3
pip install numpy==1.24.3
```

**Purpose:** Array operations, embeddings storage

**Usage:**
- Voice embedding vectors (.npy files)
- Audio waveform manipulation
- Cosine similarity calculations
- Statistical operations

---

#### 6. **SciPy** (Scientific Computing)

```python
# Version: 1.11.1
pip install scipy==1.11.1
```

**Purpose:** Signal processing, DSP operations

**Key Modules:**
- `scipy.signal`: STFT, filtering, spectral analysis
- `scipy.spatial`: Cosine similarity, distance metrics
- `scipy.stats`: Statistical analysis

**Bleep Detection:**
```python
from scipy.signal import stft, find_peaks
from scipy.spatial.distance import cosine

# STFT for spectral analysis
frequencies, times, Zxx = stft(audio, fs=16000, nperseg=1024)

# Peak detection for tone identification
peaks, _ = find_peaks(spectrum, height=threshold)
```

**Integration Points:**
- `src/show_scribe/pipelines/bleep_detection/dsp_features.py`
- `src/show_scribe/pipelines/bleep_detection/detector.py`

---

#### 7. **Librosa** (Audio Analysis)

```python
# Version: 0.10.1
pip install librosa==0.10.1
```

**Purpose:** Advanced audio feature extraction

**Features:**
- Mel spectrograms
- Chroma features
- Spectral features (centroid, bandwidth, flatness)
- Onset detection

**Bleep Detection Features:**
```python
import librosa

# Spectral centroid (frequency center of mass)
centroid = librosa.feature.spectral_centroid(y=audio, sr=16000)

# Spectral flatness (tone vs noise)
flatness = librosa.feature.spectral_flatness(y=audio)

# RMS energy
rms = librosa.feature.rms(y=audio)
```

---

### Data Management Stack

#### 8. **SQLite** (Database)

```python
# Built-in Python module
import sqlite3
```

**Purpose:** Voice bank storage, processing history

**Database File:** `voice_bank.db` (~10-100 MB)

**Key Tables:**
- `speaker_profiles`: Cast member information
- `voice_embeddings`: Voice fingerprints (links to .npy files)
- `processing_history`: Episode processing logs
- `episode_speakers`: Speaking time data
- `bleep_events`: Censorship detections
- `sfx_profiles`: Learned beep signatures

**Performance:**
- Reads: <10ms for typical queries
- Writes: <50ms with proper indexing
- Concurrent access: Single-writer, multiple-readers

**Integration Points:**
- `src/show_scribe/storage/db.py`
- `src/show_scribe/storage/schema.sql`

---

#### 9. **DuckDB** (Analytics Database) ⭐ NEW

```python
# Version: 0.9.0+
pip install duckdb>=0.9.0
```

**Purpose:** High-performance analytical queries (supplements SQLite)

**Features:**
- Column-oriented storage (OLAP workload)
- 10-100x faster than SQLite for analytics
- SQL compatible
- Zero configuration
- Works alongside SQLite
- Direct DataFrame integration

**Usage:**
```python
import duckdb

# Fast analytics queries
con = duckdb.connect('analytics.db')
result = con.execute("""
    SELECT speaker_id,
           SUM(duration_seconds) as total_time,
           COUNT(*) as segment_count
    FROM episode_speakers
    GROUP BY speaker_id
    ORDER BY total_time DESC
""").fetchdf()  # Returns pandas DataFrame
```

**Benefits:**
- Dramatically faster aggregations
- Better for complex joins
- Excellent for reporting queries
- Parquet file support

**Strategy:**
- SQLite: Transactional data (voice bank, CRUD operations)
- DuckDB: Analytics queries (aggregations, reporting)

**Integration Points:**
- `src/show_scribe/pipelines/analytics/` (all analytics modules)
- `src/show_scribe/storage/analytics_db.py`

---

#### 10. **Pandas** (Data Manipulation)

```python
# Version: 2.0.3
pip install pandas==2.0.3
```

**Purpose:** Analytics, CSV exports, data aggregation

**Usage:**
```python
import pandas as pd

# Speaking time analysis
df = pd.DataFrame(speaking_time_data)
summary = df.groupby('speaker').agg({
    'duration_seconds': 'sum',
    'word_count': 'sum',
    'segment_count': 'count'
})

# Export to CSV
df.to_csv('analytics.csv', index=False)
```

**Note:** Pandas is kept for compatibility and exports. For large dataset processing, prefer Polars (see below).

---

#### 11. **Polars** (High-Performance DataFrames) ⭐ NEW

```python
# Version: 0.19.0+
pip install polars>=0.19.0
```

**Purpose:** Fast data manipulation for large datasets

**Features:**
- 5-50x faster than pandas
- Lazy evaluation (query optimization)
- Multi-threaded by default
- Lower memory usage
- Rust-based (extremely fast)
- Pandas-compatible API

**Usage:**
```python
import polars as pl

# Much faster than pandas for large data
df = pl.read_csv("bleeps.csv")
result = (
    df.groupby("person_id")
      .agg([
          pl.col("duration_ms").sum().alias("total_ms"),
          pl.col("word_label").count().alias("bleep_count")
      ])
      .sort("total_ms", descending=True)
)

# Convert to pandas for compatibility if needed
pandas_df = result.to_pandas()
```

**When to Use:**
- Pandas: Small data (<100k rows), exports, compatibility
- Polars: Large data (>100k rows), internal processing, speed critical

**Integration Points:**
- `src/show_scribe/pipelines/analytics/` (for large datasets)
- `src/show_scribe/utils/data_processing.py`

---

### User Interface Stack

#### 12. **Streamlit** (Web UI)

```python
# Version: 1.28.0
pip install streamlit==1.28.0
```

**Purpose:** Review UI for unknown speakers and bleeps

**Features:**
- Audio player component
- Data tables with filtering
- Interactive forms
- Real-time updates
- Session state management

**Key Components:**
```python
import streamlit as st

# Audio playback
st.audio(audio_file, format='audio/wav')

# Data table
st.dataframe(bleeps_df, use_container_width=True)

# Edit form
with st.form('bleep_review'):
    word = st.text_input('Word Label', value='[BLEEP]')
    st.form_submit_button('Save')
```

**Integration Points:**
- `ui/streamlit_app/app.py`
- `ui/streamlit_app/pages/`
- `ui/streamlit_app/components/`

---

### Utilities & Helpers

#### 11. **pydantic** (Data Validation)

```python
# Version: 2.4.2
pip install pydantic==2.4.2
```

**Purpose:** Configuration validation, data models

**Example:**
```python
from pydantic import BaseModel, Field

class ShowConfig(BaseModel):
    show_name: str
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    default_speakers: list[str] = []
```

---

#### 12. **PyYAML** (Configuration)

```python
# Version: 6.0.1
pip install pyyaml==6.0.1
```

**Purpose:** YAML config file parsing

---

#### 14. **Typer** (CLI Framework) ⭐ UPGRADED

```python
# Version: 0.9.0+
pip install typer[all]>=0.9.0
```

**Purpose:** Modern CLI interface with type hints and validation

**Features:**
- Type hints for automatic validation
- Auto-generated help text
- Shell completion (bash/zsh/fish)
- Rich integration for beautiful output
- Less boilerplate than argparse
- Better error messages

**Usage:**
```python
import typer
from rich.console import Console

app = typer.Typer()
console = Console()

@app.command()
def process(
    video_path: str = typer.Argument(..., help="Path to video file"),
    show_name: str = typer.Option(None, "--show", "-s"),
    resume: bool = typer.Option(False, "--resume", "-r")
):
    """Process a TV show episode."""
    console.print(f"[bold green]Processing:[/bold green] {video_path}")
    # Processing logic...

if __name__ == "__main__":
    app()
```

**Benefits:**
- Much better developer experience than argparse
- Automatic validation from type hints
- Beautiful help text
- Shell completion out of the box

**Integration Points:**
- `src/show_scribe/cli.py`

---

#### 15. **Rich** (Terminal Output) ⭐ UPGRADED

```python
# Version: 13.7.0+
pip install rich>=13.7.0
```

**Purpose:** Beautiful terminal output and progress bars

**Features:**
- Rich text formatting (colors, styles, emoji)
- Multiple progress bars
- Tables with automatic formatting
- Syntax highlighting
- Tree views
- Live displays

**Usage:**
```python
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeRemainingColumn
from rich.console import Console
from rich.table import Table

console = Console()

# Beautiful progress bars
with Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TimeRemainingColumn(),
) as progress:
    task = progress.add_task("[cyan]Transcribing...", total=100)
    progress.update(task, advance=10)

# Beautiful tables
table = Table(title="Episode Analytics")
table.add_column("Speaker", style="cyan")
table.add_column("Duration", justify="right", style="green")
table.add_row("Michael Scott", "25:30")
console.print(table)
```

**Benefits:**
- Much better UX than tqdm
- Consistent styling across application
- More features (tables, trees, syntax highlighting)

**Integration Points:**
- `src/show_scribe/cli.py`
- `src/show_scribe/utils/logging.py`
- All progress indicators

---

## Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   CLI Tool   │  │  Streamlit   │  │  (Future:    │     │
│  │ (show-scribe)│  │     UI       │  │   Web App)   │     │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘     │
└─────────┼──────────────────┼──────────────────────────────┘
          │                  │
          ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                      Pipeline Orchestrator                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Stage Manager (Checkpoints, Resume, Error Recovery)│   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                     Processing Stages                        │
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Audio      │───▶│     ASR      │───▶│ Diarization  │ │
│  │  Extraction  │    │  (Whisper)   │    │  (Pyannote)  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                                         │          │
│         ▼                                         ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │    Bleep     │    │   Speaker    │◀───│    Voice     │ │
│  │  Detection   │    │     ID       │    │ Embeddings   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                    │                              │
│         └────────┬───────────┘                              │
│                  ▼                                           │
│         ┌──────────────┐    ┌──────────────┐              │
│         │  Transcript  │    │  Analytics   │              │
│         │  Generator   │    │  Calculator  │              │
│         └──────────────┘    └──────────────┘              │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                      Storage Layer                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   SQLite     │    │    File      │    │    Voice     │ │
│  │   Database   │    │   System     │    │     Bank     │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Video File (.mp4)
    ↓
[FFmpeg] Extract Audio
    ↓
Audio File (.wav, 16kHz mono)
    ↓
[Whisper] Transcribe ──────────→ transcript_raw.json
    ↓                              (timestamps, text, confidence)
    ├──→ [Pyannote] Diarize ───→ diarization.json
    │       ↓                     (speaker segments)
    │       │
    ├──→ [DSP] Detect Bleeps ──→ bleeps_detected.json
    │       ↓                     (tone/mute/noise events)
    │       │
    ↓       ↓
[Alignment] Match ASR + Diarization + Bleeps
    ↓
[Embeddings] Generate speaker fingerprints
    ↓
[Voice Bank] Match against known speakers
    ↓
    ├──→ High confidence (≥0.85) ──→ Auto-label
    ├──→ Medium confidence (0.70-0.85) ──→ Flag for review
    └──→ Low confidence (<0.70) ──→ Require manual label
    ↓
[User Review] Label unknowns + review bleeps
    ↓
[Transcript Builder] Generate final outputs
    ↓
    ├──→ transcript_final.txt (plain text)
    ├──→ transcript_final.srt (subtitles)
    ├──→ transcript_final.json (structured)
    ├──→ bleeps.csv (censorship table)
    ├──→ bleeps.json (detailed)
    └──→ analytics.json (stats)

Use ``render_plain_text_with_alignment`` / ``render_srt_with_alignment`` when you want inline
confidence + word timelines. Call the base ``render_plain_text`` / ``render_srt`` functions for
the original single-line-per-segment output.
```

### Module Dependencies

```
show_scribe/
├── cli.py
│   └── uses: pipelines/*, storage/*, utils/*
│
├── pipelines/
│   ├── extract_audio.py
│   │   └── uses: utils/ffmpeg, utils/audio_io
│   ├── asr/
│   │   └── uses: whisper, utils/audio_io
│   ├── diarization/
│   │   └── uses: pyannote.audio, utils/audio_io
│   ├── embeddings/
│   │   └── uses: resemblyzer, numpy, storage/paths
│   ├── bleep_detection/
│   │   └── uses: numpy, scipy, librosa, storage/db
│   ├── speaker_id/
│   │   └── uses: embeddings/*, storage/voice_bank, numpy
│   ├── alignment/
│   │   └── uses: diarization/*, asr/*, bleep_detection/*
│   ├── transcript/
│   │   └── uses: alignment/*, storage/paths
│   └── analytics/
│       └── uses: storage/db, pandas
│
├── storage/
│   ├── db.py
│   │   └── uses: sqlite3, pydantic
│   ├── voice_bank.py
│   │   └── uses: db, numpy, embeddings/*
│   └── paths.py
│       └── uses: pathlib, yaml
│
└── utils/
    ├── ffmpeg.py
    ├── audio_io.py
    ├── timecode.py
    └── logging.py
```

---

## Development Stack

### Code Quality Tools

#### **Ruff** (Linter)

```toml
# ruff.toml
target-version = "py311"
line-length = 100
select = ["E", "F", "I", "N", "W", "B", "UP"]
ignore = ["E501"]  # Line too long (handled by formatter)

[per-file-ignores]
"__init__.py" = ["F401"]  # Unused imports
"tests/*" = ["S101"]  # Use of assert
```

**Installation:**
```bash
pip install ruff==0.1.3
```

---

#### **Black** (Formatter)

```toml
# pyproject.toml
[tool.black]
line-length = 100
target-version = ['py311']
include = '\.pyi?$'
```

**Installation:**
```bash
pip install black==23.9.1
```

---

#### **mypy** (Type Checker)

```ini
# mypy.ini
[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
```

**Installation:**
```bash
pip install mypy==1.5.1
```

---

#### **pytest** (Testing)

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=src/show_scribe --cov-report=html"
```

**Installation:**
```bash
pip install pytest==7.4.2
pip install pytest-cov==4.1.0
pip install pytest-asyncio==0.21.1
```

---

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.3
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

**Installation:**
```bash
pip install pre-commit==3.4.0
pre-commit install
```

---

### CI/CD (GitHub Actions)

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3

      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          brew install ffmpeg
          pip install -e ".[dev]"

      - name: Lint
        run: |
          ruff check .
          black --check .

      - name: Type check
        run: mypy src/

      - name: Test
        run: pytest
```

---

## Performance Considerations

### Optimization Strategies

#### 1. **Parallel Processing**

```python
from concurrent.futures import ProcessPoolExecutor

def process_episodes_parallel(episodes, max_workers=2):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_episode, ep) for ep in episodes]
        results = [f.result() for f in futures]
    return results
```

**Note:** Limit to 1-2 concurrent episodes to avoid memory issues.

---

#### 2. **Model Caching**

```python
# Cache loaded models globally
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("large-v3")
    return _whisper_model
```

---

#### 3. **Chunked Processing**

```python
# For very long episodes (>2 hours)
def process_long_episode(audio_file, chunk_duration=1800):  # 30 min chunks
    chunks = split_audio(audio_file, chunk_duration)
    results = [process_chunk(chunk) for chunk in chunks]
    return merge_results(results)
```

---

#### 4. **Database Indexing**

```sql
-- Performance-critical indexes
CREATE INDEX idx_bleep_episode_time ON bleep_events(episode_id, start_ms);
CREATE INDEX idx_bleep_person ON bleep_events(person_id);
CREATE INDEX idx_embedding_speaker ON voice_embeddings(speaker_id);
CREATE INDEX idx_episode_speaker ON episode_speakers(episode_id, speaker_id);
```

---

#### 5. **Memory Management**

```python
import gc

def process_with_cleanup(audio_file):
    try:
        result = process_audio(audio_file)
        return result
    finally:
        gc.collect()  # Force garbage collection
```

---

### Performance Benchmarks

| Hardware | ASR (1hr) | Diarization | Bleep Detection | Total |
|----------|-----------|-------------|-----------------|-------|
| M1 Mac (API) | 10 min | 45 min | 3 min | **~1 hour** |
| M1 Mac (Local) | 45 min | 45 min | 3 min | **~1.5 hours** |
| M3 Max (API) | 8 min | 30 min | 2 min | **~40 min** |
| M3 Max (Local) | 25 min | 30 min | 2 min | **~1 hour** |
| Intel i7 (API) | 12 min | 60 min | 4 min | **~1.25 hours** |
| Intel i7 (Local) | 90 min | 60 min | 4 min | **~2.5 hours** |

---

## Security & Privacy

### Data Protection

1. **Local Processing**
   - All voice data stays on device when using local models
   - No network calls except when using API mode
   - Voice bank encrypted at rest (future enhancement)

2. **API Mode**
   - OpenAI Whisper API: Audio sent to OpenAI servers
   - Data retention: 30 days (OpenAI policy)
   - User should be informed of data transmission

3. **Voice Biometrics**
   - Voice embeddings are mathematical representations, not raw audio
   - Embeddings stored as .npy files (binary, not easily reconstructed)
   - Database contains no PII beyond user-provided names

4. **Access Control**
   - SQLite database uses file system permissions
   - No authentication system (single-user app)
   - Future: Multi-user access control if needed

---

## Alternative Technologies Considered

### ASR Alternatives

| Technology | Pros | Cons | Decision |
|------------|------|------|----------|
| **Whisper** ✅ | Best accuracy, multilingual, open source | Slower than real-time | **Selected** |
| AssemblyAI | Fast, good accuracy, diarization included | Expensive ($0.15/hour), cloud-only | Alternative option |
| Google Speech-to-Text | Fast, reliable | Less accurate on diverse audio, expensive | Not selected |
| Rev.ai | Human-level accuracy | Very expensive, slow turnaround | Not selected |

---

### Diarization Alternatives

| Technology | Pros | Cons | Decision |
|------------|------|------|----------|
| **Pyannote** ✅ | SOTA, open source, well-maintained | Requires HuggingFace token | **Selected** |
| AssemblyAI | Integrated with ASR | Expensive, cloud-only | Alternative |
| AWS Transcribe | Built-in diarization | Less accurate, vendor lock-in | Not selected |
| Custom Model | Full control | High development cost | Future consideration |

---

### Embedding Alternatives

| Technology | Pros | Cons | Decision |
|------------|------|------|----------|
| **Resemblyzer** ✅ | Simple API, good performance | Older model | **Selected** |
| Pyannote Embeddings | Integrated with diarization | More complex setup | Alternative (also supported) |
| SpeakerNet | High accuracy | Requires training | Not selected |
| X-Vectors | Industry standard | Complex implementation | Future enhancement |

---

### UI Alternatives

| Technology | Pros | Cons | Decision |
|------------|------|------|----------|
| **Streamlit** ✅ | Fast development, Python-native | Limited customization | **Selected for MVP** |
| Electron + React | Native feel, highly customizable | Requires JS knowledge | Future enhancement |
| SwiftUI | Best macOS integration | Swift only, steep learning curve | Future consideration |
| Flask + HTML/CSS | Full control | More development time | Not selected |

---

## Integration Specifications

### OpenAI Whisper API

```python
import openai

openai.api_key = "sk-..."

def transcribe_with_api(audio_file):
    with open(audio_file, "rb") as f:
        transcript = openai.Audio.transcribe(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"]
        )
    return transcript
```

**Cost:** $0.006 per minute
**Rate Limit:** 50 requests per minute
**Max File Size:** 25 MB

---

### Pyannote.audio HuggingFace Integration

```python
from pyannote.audio import Pipeline

# Requires HuggingFace account and token
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1",
    use_auth_token="hf_..."
)
```

**Setup:**
1. Create HuggingFace account
2. Accept model license agreement
3. Generate access token
4. Configure in app settings

---

### FFmpeg Integration

```python
import ffmpeg

def extract_audio(video_path, output_path):
    stream = ffmpeg.input(video_path)
    stream = ffmpeg.output(
        stream,
        output_path,
        acodec='pcm_s16le',
        ar=16000,
        ac=1,
        af='loudnorm=I=-20:TP=-1.5'
    )
    ffmpeg.run(stream, overwrite_output=True, quiet=True)
```

---

## Deployment & Distribution

### Installation Methods

#### 1. **pip Install (Recommended for developers)**

```bash
pip install show-scribe
```

#### 2. **Homebrew (Planned)**

```bash
brew install show-scribe
```

#### 3. **macOS .app Bundle (Future)**

Packaged with:
- py2app or PyInstaller
- Bundled Python runtime
- Pre-downloaded models
- Double-click to run

---

### Packaging Configuration

```toml
# pyproject.toml
[project]
name = "show-scribe"
version = "1.0.0"
description = "Automated voice identification and transcription for TV episodes"
authors = [{name = "Your Name", email = "email@example.com"}]
requires-python = ">=3.11,<3.12"
dependencies = [
    "openai-whisper==20231117",
    "pyannote.audio==3.1.1",
    "resemblyzer==0.1.1",
    "ffmpeg-python==0.2.0",
    "numpy==1.24.3",
    "scipy==1.11.1",
    "librosa==0.10.1",
    "streamlit==1.28.0",
    "pandas==2.0.3",
    "pydantic==2.4.2",
    "pyyaml==6.0.1",
    "tqdm==4.66.1",
]

[project.optional-dependencies]
dev = [
    "pytest==7.4.2",
    "pytest-cov==4.1.0",
    "ruff==0.1.3",
    "black==23.9.1",
    "mypy==1.5.1",
    "pre-commit==3.4.0",
]

[project.scripts]
show-scribe = "show_scribe.cli:main"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

---

### Distribution Checklist

- [ ] PyPI package published
- [ ] Homebrew formula created
- [ ] Documentation website hosted
- [ ] GitHub releases automated
- [ ] Installation guide tested on clean macOS
- [ ] CI/CD pipeline validated
- [ ] License file included
- [ ] CHANGELOG maintained

---

## System Requirements Summary

### Minimum Configuration

```
macOS: 12.0 (Monterey)
CPU: Apple M1 or Intel i5 (2015+)
RAM: 8 GB
Storage: 10 GB free
Internet: Required for model downloads, optional for API mode
```

### Recommended Configuration

```
macOS: 14.0 (Sonoma)
CPU: Apple M2 or M3
RAM: 16 GB
Storage: 50 GB free (for multiple shows)
SSD: Required for best performance
Internet: High-speed for API mode
```

### Storage Requirements

| Component | Size |
|-----------|------|
| Application Code | ~50 MB |
| Whisper large-v3 | 2.9 GB |
| Pyannote Models | ~350 MB |
| Resemblyzer Model | ~83 MB |
| Voice Bank (per show) | ~50-500 MB |
| Processed Episode | ~500 MB - 2 GB |
| **Total Install** | **~3.5 GB** |

---

## Conclusion

Show-Scribe leverages best-in-class open-source technologies to deliver accurate, automated speaker identification and censorship detection. The architecture is designed for:

- **Performance:** Optimized for Apple Silicon, with local and cloud options
- **Accuracy:** Industry-leading models (Whisper, Pyannote)
- **Extensibility:** Modular design, easy to add features
- **Privacy:** Local processing option, minimal data transmission
- **Maintainability:** Type-checked Python, comprehensive tests

**Technology Stack Status:** ✅ Production Ready

---

**Document Version:** 1.0
**Last Updated:** October 16, 2025
**Next Review:** Q1 2026
