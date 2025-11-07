# Show-Scribe Solution Architecture

**Version:** 1.0
**Date:** October 16, 2025
**Status:** Implementation Ready
**Classification:** Technical Architecture Document

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architectural Vision & Principles](#architectural-vision--principles)
3. [System Context](#system-context)
4. [Component Architecture](#component-architecture)
5. [Data Architecture](#data-architecture)
6. [Integration Architecture](#integration-architecture)
7. [Security Architecture](#security-architecture)
8. [Performance Architecture](#performance-architecture)
9. [Deployment Architecture](#deployment-architecture)
10. [Reliability & Availability](#reliability--availability)
11. [Monitoring & Observability](#monitoring--observability)
12. [Scalability Roadmap](#scalability-roadmap)
13. [Technology Decision Records](#technology-decision-records)
14. [Architecture Evolution Plan](#architecture-evolution-plan)

---

## Executive Summary

### Purpose

Show-Scribe is an intelligent audio analysis system designed to automatically transcribe TV show episodes with speaker identification and censorship detection. This document defines the comprehensive solution architecture that enables accurate, scalable, and maintainable processing of audio content.

### Key Architectural Goals

| Goal | Description | Success Metric |
|------|-------------|----------------|
| **Accuracy** | Deliver industry-leading transcription and speaker identification | >95% speaker ID accuracy after 5 episodes |
| **Performance** | Process 1-hour episodes efficiently | <90 min total processing time |
| **Reliability** | Ensure consistent processing with fault tolerance | 99% successful processing rate |
| **Scalability** | Support growing episode libraries and voice banks | Handle 1000+ episodes per show |
| **Maintainability** | Enable rapid feature development and bug fixes | <2 hours for hotfix deployment |
| **Privacy** | Protect sensitive voice biometric data | Zero unauthorized data access |

### Architectural Approach

Show-Scribe employs a **modular pipeline architecture** with:
- **Sequential stage processing** with checkpoints for fault tolerance
- **Local-first design** with optional cloud enhancement
- **Persistent learning** through voice bank accumulation
- **Extensible plugin system** for swappable components
- **Configuration-driven behavior** for customization without code changes

---

## Architectural Vision & Principles

### Core Architectural Principles

#### 1. **Modularity First**
- Each pipeline stage is independently testable and replaceable
- Clear interfaces between components enable parallel development
- Component coupling is minimized through well-defined contracts

#### 2. **Local-First with Cloud-Optional**
- Default operation requires no internet connection
- Cloud services enhance (not enable) core functionality
- User maintains full control over data sovereignty

#### 3. **Progressive Learning**
- System accuracy improves with each processed episode
- Voice bank grows organically without manual corpus creation
- Confidence-based automation reduces user intervention over time

#### 4. **Fail-Safe Operations**
- Checkpoint system enables recovery from any stage failure
- Partial results are always preserved
- User never loses work due to crashes or interruptions

#### 5. **Extensibility by Design**
- Plugin architecture for new providers (ASR, diarization, embeddings)
- Configuration-driven behavior enables customization without code changes
- Clear extension points for future features (emotion detection, topic modeling)

#### 6. **Performance Without Compromise**
- Leverage hardware acceleration (Apple Neural Engine, GPU)
- Intelligent caching reduces redundant computation
- Parallel processing where possible, sequential where necessary

#### 7. **Data Privacy & Security**
- Minimize data transmission (local processing preferred)
- Encryption at rest for sensitive voice biometrics (future)
- Clear user consent for any cloud operations

#### 8. **Observable & Debuggable**
- Comprehensive logging at appropriate verbosity levels
- Metrics collection for performance monitoring
- Clear error messages with actionable recovery steps

---

## System Context

### System Context Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          External Context                            │
│                                                                       │
│  ┌──────────────┐         ┌──────────────┐         ┌─────────────┐ │
│  │   End User   │────────▶│ Show-Scribe  │◀────────│Media Library│ │
│  │ (Producer/   │         │    System    │         │ (Video Files│ │
│  │  Analyst)    │         │              │         │  on Disk)   │ │
│  └──────────────┘         └──────┬───────┘         └─────────────┘ │
│                                   │                                  │
│                                   │                                  │
│        ┌──────────────────────────┼──────────────────────────┐     │
│        │                          │                          │     │
│        ▼                          ▼                          ▼     │
│  ┌─────────────┐          ┌─────────────┐          ┌─────────────┐ │
│  │  OpenAI     │          │ HuggingFace │          │  ElevenLabs │ │
│  │  Whisper    │          │  (Pyannote  │          │  (Optional  │ │
│  │    API      │          │   Models)   │          │    ASR)     │ │
│  │ (Optional)  │          │             │          │             │ │
│  └─────────────┘          └─────────────┘          └─────────────┘ │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Stakeholders

| Stakeholder | Role | Primary Concerns |
|-------------|------|------------------|
| **Content Producers** | Primary users who process episodes | Accuracy, speed, ease of use |
| **Media Analysts** | Review transcripts and analytics | Data quality, export formats |
| **Research Teams** | Use voice bank for studies | Privacy, data access controls |
| **IT Operations** | Deploy and maintain system | Reliability, monitoring, backups |
| **Development Team** | Build and enhance features | Code maintainability, test coverage |
| **Compliance Officers** | Ensure data privacy compliance | Security, audit trails, data retention |

### System Boundaries

**In Scope:**
- Audio extraction from video files
- Speech-to-text transcription
- Speaker diarization and identification
- Censorship/bleep detection
- Voice bank management
- Transcript generation and export
- Speaking time analytics
- Review and labeling UI

**Out of Scope (Current Version):**
- Video processing beyond audio extraction
- Real-time streaming transcription
- Multi-language translation
- Emotion/sentiment analysis (future roadmap)
- Cloud-based collaboration (future roadmap)
- Mobile applications

---

## Component Architecture

### High-Level Component View

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │   CLI Interface │  │  Streamlit UI   │  │  REST API       │    │
│  │  (show-scribe)  │  │  (Review Tool)  │  │  (Future)       │    │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘    │
└───────────┼─────────────────────┼───────────────────────┐
            │                       │                          │
            ▼                       ▼                          ▼
┌───────────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                                │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │              Pipeline Orchestrator                            │    │
│  │  ┌──────────────────────────────────────────────────────┐   │    │
│  │  │  Stage Manager (Checkpoints, Resume, Error Recovery)│   │    │
│  │  └──────────────────────────────────────────────────────┘   │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                        │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐       │
│  │   Audio    │ │    ASR     │ │ Diarization│ │   Bleep    │       │
│  │ Extraction │ │ (Whisper)  │ │ (Pyannote) │ │ Detection  │       │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘       │
│                                                                        │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐       │
│  │  Speaker   │ │   Voice    │ │ Transcript │ │ Analytics  │       │
│  │     ID     │ │ Embeddings │ │  Builder   │ │ Calculator │       │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘       │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│                       STORAGE LAYER                                   │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │
│  │    SQLite      │  │   File System  │  │   Voice Bank   │         │
│  │   Database     │  │   (Episodes,   │  │  (Embeddings,  │         │
│  │  (Metadata)    │  │   Audio, etc)  │  │    Samples)    │         │
│  └────────────────┘  └────────────────┘  └────────────────┘         │
└───────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

#### Presentation Layer

**CLI Interface (`cli.py`)**
- Command-line entry point for batch processing
- Argument parsing and validation
- Progress reporting
- Error handling and user feedback

**Streamlit UI**
- Interactive review interface for unknown speakers
- Bleep event review and word labeling
- Voice bank management UI
- Analytics dashboard
- Audio playback with waveform visualization

**REST API (Future)**
- HTTP endpoints for remote control
- Integration with other tools
- Webhook notifications

#### Application Layer

**Pipeline Orchestrator**
- Coordinates execution of processing stages
- Manages stage dependencies
- Handles stage failures and retries
- Provides progress tracking

**Stage Manager**
- Creates checkpoints after each stage
- Enables resume from last successful checkpoint
- Cleans up temporary files
- Validates stage outputs before proceeding

**Audio Extraction (`extract_audio.py`)**
- Converts video to audio using FFmpeg
- Normalizes audio levels (-20dB LUFS)
- Resamples to 16kHz mono
- Validates audio quality

**ASR Pipeline (`asr/`)**
- Whisper integration (local or API)
- Word-level timestamp generation
- Confidence score extraction
- Multi-language support

**Diarization Pipeline (`diarization/`)**
- Pyannote speaker segmentation
- Voice activity detection
- Speaker change point detection
- Overlapping speech handling

**Bleep Detection (`bleep_detection/`)**
- DSP-based feature extraction
- Tone/mute/noise classification
- Event merging and filtering
- Speaker attribution

**Speaker Identification (`speaker_id/`)**
- Voice embedding generation
- Cosine similarity matching
- Confidence-based labeling
- Voice bank CRUD operations

**Transcript Builder (`transcript/`)**
- Merges ASR + diarization + bleeps
- Generates speaker-labeled text
- Exports to multiple formats (TXT, SRT, JSON)
- Inserts [BLEEP] tokens

**Analytics Calculator (`analytics/`)**
- Speaking time per speaker
- Word count aggregation
- Bleep statistics
- Cross-episode trends

#### Storage Layer

**SQLite Database**
- Speaker profiles
- Voice embeddings metadata
- Processing history
- Episode-speaker relationships
- Bleep events
- SFX profiles

**File System**
- Episode directory structure
- Audio files (extracted WAV)
- Transcript outputs
- Voice embeddings (.npy files)
- Audio samples for reference

**Voice Bank**
- Persistent speaker profiles
- Embedding storage and retrieval
- Backup and restore operations
- Import/export functionality

### Component Interaction Patterns

**Sequential Pipeline Pattern**
```
Stage 1 → Checkpoint → Stage 2 → Checkpoint → Stage 3 → ...
```

**Service-Based Pattern**
```
Pipeline Stage → Service Interface → Provider Implementation
                                          ↓
                                    (Whisper / ElevenLabs / etc)
```

**Repository Pattern**
```
Business Logic → Repository Interface → Database / File System
```

**Observer Pattern**
```
Pipeline Stage → Event Emitter → Progress Listeners
                                       ↓
                                  (CLI / UI / Logs)
```

---

## Data Architecture

### Data Model

#### Conceptual Data Model

```
┌─────────────────┐
│   Show          │
│  - name         │
│  - config       │
└────────┬────────┘
         │ 1
         │
         │ *
┌────────▼────────┐
│   Episode       │
│  - id           │
│  - path         │
│  - duration     │
└────────┬────────┘
         │ 1
         │
         │ *
┌────────▼────────┐       ┌─────────────────┐
│   Segment       │──────▶│  Speaker        │
│  - start        │   *   │  - name         │
│  - end          │  ────▶│  - profile_id   │
│  - text         │   1   └────────┬────────┘
│  - speaker      │                │ 1
└────────┬────────┘                │
         │                         │ *
         │                ┌────────▼────────┐
         │                │  Embedding      │
         │                │  - vector       │
         │                │  - quality      │
         │                └─────────────────┘
         │ 1
         │
         │ *
┌────────▼────────┐
│   BleepEvent    │
│  - start        │
│  - type         │
│  - word_label   │
│  - speaker      │
└─────────────────┘
```

#### Physical Data Model

**Database: `voice_bank.db`**

Primary Tables:
- `speaker_profiles`: 10-100 rows per show
- `voice_embeddings`: 100-2000 rows per show
- `processing_history`: 1 row per episode
- `episode_speakers`: 5-20 rows per episode
- `bleep_events`: 0-50 rows per episode
- `sfx_profiles`: 5-20 rows (learned signatures)

**File System Storage**

```
~/Documents/VoiceTranscriptTool/
├── voice_bank/
│   ├── voice_bank.db (10-100 MB)
│   ├── embeddings/
│   │   └── speaker_<id>_<timestamp>.npy (17 KB each)
│   └── audio_samples/
│       └── speaker_<id>_<timestamp>.wav (100-500 KB each)
└── shows/
    └── [ShowName]/
        ├── show_config.json (5 KB)
        └── episodes/
            └── S01E01/
                ├── metadata.json (10 KB)
                ├── audio_extracted.wav (600-800 MB per hour)
                ├── transcript_raw.json (200-500 KB)
                ├── diarization.json (50-100 KB)
                ├── transcript_final.txt (100-300 KB)
                ├── transcript_final.srt (100-300 KB)
                ├── transcript_final.json (300-800 KB)
                ├── analytics.json (20 KB)
                ├── bleeps.csv (5-20 KB)
                ├── bleeps.json (10-30 KB)
                └── checkpoints/ (checkpoint flags)
```

### Data Flow Architecture

**Processing Data Flow**

```
Input Video (MP4)
    ↓ [FFmpeg]
Audio WAV (16kHz mono) ────┐
    ↓ [Whisper]            │
transcript_raw.json        │
    ↓                      │
    ├─────────────────────┘
    │ [Pyannote]
    ↓
diarization.json
    ↓
    ├────[DSP Analysis]───▶ bleeps_detected.json
    │
    ↓ [Alignment]
aligned_segments.json
    ↓ [Embeddings]
speaker_embeddings/
    ↓ [Voice Bank Match]
labeled_segments.json
    ↓ [Review UI]
confirmed_labels.json
    ↓ [Transcript Builder]
├─▶ transcript_final.txt
├─▶ transcript_final.srt
├─▶ transcript_final.json
├─▶ bleeps.csv
├─▶ bleeps.json
└─▶ analytics.json
```

### Data Retention Policy

| Data Type | Retention | Rationale |
|-----------|-----------|-----------|
| Extracted audio WAV | 30 days | Can be re-extracted from source |
| Transcript outputs | Permanent | Primary deliverable |
| Voice embeddings | Permanent | Required for future episodes |
| Intermediate JSON | 7 days | Debugging purposes |
| Checkpoint flags | Until next successful run | Resume capability |
| Processing logs | 90 days | Troubleshooting |
| Audio samples | Permanent | Voice bank references |

### Data Backup Strategy

**Automated Backups:**
- Voice bank database: Daily (7-day retention)
- Configuration files: Weekly (30-day retention)
- Transcript outputs: User-managed

**Manual Exports:**
- Voice bank export: ZIP archive with database + embeddings
- Episode bundle: All outputs for single episode

---

## Integration Architecture

### External Service Integrations

#### OpenAI Whisper API

**Integration Type:** REST API
**Purpose:** Cloud-based ASR (alternative to local Whisper)
**Protocol:** HTTPS
**Authentication:** API Key

**Integration Points:**
```python
# src/show_scribe/pipelines/asr/whisper_api.py

import openai

def transcribe_with_api(audio_file_path: str, api_key: str) -> dict:
    """
    Transcribe audio using OpenAI Whisper API.

    Returns:
        {
            "text": "full transcript",
            "segments": [...],
            "language": "en"
        }
    """
    with open(audio_file_path, "rb") as f:
        response = openai.Audio.transcribe(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"]
        )
    return response
```

**Error Handling:**
- Retry on 429 (rate limit): Exponential backoff
- Fallback to local Whisper on persistent failures
- Timeout: 300 seconds

**Configuration:**
```yaml
whisper:
  use_local: false
  api_key: ${OPENAI_API_KEY}
  model: "whisper-1"
  timeout_seconds: 300
  max_retries: 3
```

#### HuggingFace (Pyannote Models)

**Integration Type:** Model Download
**Purpose:** Diarization model weights
**Protocol:** HTTPS
**Authentication:** HuggingFace Token

**Integration Points:**
```python
# src/show_scribe/pipelines/diarization/pyannote_pipeline.py

from pyannote.audio import Pipeline

def load_diarization_pipeline(hf_token: str) -> Pipeline:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1",
        use_auth_token=hf_token
    )
    return pipeline
```

**Setup Requirements:**
1. User creates HuggingFace account
2. Accepts Pyannote model license
3. Generates access token
4. Configures in app settings

#### FFmpeg (System Dependency)

**Integration Type:** System Binary
**Purpose:** Audio/video processing
**Protocol:** Command-line execution

**Integration Points:**
```python
# src/show_scribe/utils/ffmpeg.py

import ffmpeg

def extract_audio(video_path: str, output_path: str) -> None:
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

**Verification:**
```bash
# scripts/verify_dependencies.py
ffmpeg -version
```

### Integration Patterns

**Provider Pattern**
```python
# Abstract interface
class ASRProvider(Protocol):
    def transcribe(self, audio_path: str) -> dict:
        ...

# Concrete implementations
class WhisperLocalProvider(ASRProvider):
    def transcribe(self, audio_path: str) -> dict:
        return whisper.load_model("large-v3").transcribe(audio_path)

class WhisperAPIProvider(ASRProvider):
    def transcribe(self, audio_path: str) -> dict:
        return openai.Audio.transcribe(...)

# Factory
def get_asr_provider(config: dict) -> ASRProvider:
    if config["use_local"]:
        return WhisperLocalProvider()
    else:
        return WhisperAPIProvider(api_key=config["api_key"])
```

---

## Security Architecture

### Threat Model

| Threat | Impact | Likelihood | Mitigation |
|--------|--------|------------|------------|
| Unauthorized voice bank access | High | Medium | File system permissions, future encryption |
| API key exposure | Medium | High | Environment variables, .env exclusion |
| Man-in-the-middle (API calls) | Medium | Low | HTTPS only, certificate validation |
| Data exfiltration (cloud mode) | High | Low | User consent, clear privacy policy |
| Malicious video files | Low | Medium | Input validation, sandboxed FFmpeg |
| Database corruption | Medium | Low | Automated backups, transaction integrity |

### Security Controls

#### Authentication & Authorization

**Current (v1.0):**
- Single-user application (no auth required)
- File system permissions control access
- API keys stored in environment variables

**Future (v2.0):**
- Multi-user support with role-based access
- OAuth2 for cloud integrations
- Voice bank access control lists

#### Data Protection

**At Rest:**
- Voice embeddings stored as binary .npy files (not easily reconstructed)
- SQLite database uses file system permissions
- API keys in .env files (git-ignored)

**Future Enhancements:**
- SQLite encryption using SQLCipher
- Voice embedding encryption with user-provided key
- Secure enclave for API credentials (macOS Keychain)

**In Transit:**
- HTTPS for all API calls (OpenAI, HuggingFace)
- No unencrypted transmission of audio data
- Certificate pinning for critical endpoints

#### Input Validation

```python
# Validate video file before processing
def validate_video_input(file_path: str) -> bool:
    # Check file existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")

    # Check file extension
    valid_extensions = ['.mp4', '.mkv', '.avi', '.mov']
    if not any(file_path.endswith(ext) for ext in valid_extensions):
        raise ValueError("Unsupported file format")

    # Check file size (max 10 GB)
    file_size = os.path.getsize(file_path)
    if file_size > 10 * 1024 * 1024 * 1024:
        raise ValueError("File too large (max 10 GB)")

    return True
```

#### Audit Logging

```python
# Log all significant operations
logging.info(f"Processing started: episode={episode_id}, user={user}, timestamp={now()}")
logging.info(f"Voice bank updated: speaker_id={speaker_id}, action=add_embedding")
logging.warning(f"API call failed: provider=OpenAI, error={error_message}")
```

### Privacy Considerations

**Data Minimization:**
- Only extract audio (no video processing/storage)
- Voice embeddings are mathematical representations (not raw audio)
- No PII collected beyond user-provided speaker names

**User Control:**
- Clear indication when cloud services are used
- Option to process fully locally
- Voice bank export/delete capabilities

**Compliance:**
- GDPR considerations: Right to erasure (delete voice bank)
- Data portability: Export voice bank in standard format
- Consent: User explicitly enables cloud mode

---

## Performance Architecture

### Performance Requirements

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Episode processing time (1hr, API) | <30 min | End-to-end timer |
| Episode processing time (1hr, local) | <2 hours | End-to-end timer |
| UI response time | <500ms | HTTP request latency |
| Database query time | <50ms | SQLite profiler |
| Memory usage (peak) | <6 GB | System monitor |
| Disk I/O (peak) | <100 MB/s | iostat |

### Performance Optimization Strategies

#### 1. Hardware Acceleration

**Apple Neural Engine (ANE):**
```python
# Whisper on Apple Silicon with CoreML
import coremltools as ct

# Convert Whisper model to CoreML format
coreml_model = ct.convert(whisper_model, compute_units=ct.ComputeUnit.ALL)

# This enables ANE acceleration automatically
```

**GPU Acceleration (CUDA/Metal):**
```python
# Pyannote with GPU
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1",
    use_auth_token=hf_token
).to(torch.device("mps"))  # Metal Performance Shaders on macOS
```

#### 2. Caching Strategy

**Model Caching:**
```python
# Global model cache to avoid reloading
_model_cache = {}

def get_cached_model(model_name: str):
    if model_name not in _model_cache:
        _model_cache[model_name] = load_model(model_name)
    return _model_cache[model_name]
```

**Intermediate Results Caching:**
- Cache transcript_raw.json to avoid re-transcribing
- Cache diarization.json for re-processing
- Cache embeddings to avoid regeneration

#### 3. Parallel Processing

**Batch Episode Processing:**
```python
from concurrent.futures import ProcessPoolExecutor

def process_episodes_parallel(episodes: list[str], max_workers: int = 2):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_episode, ep) for ep in episodes]
        results = [f.result() for f in futures]
    return results
```

**Note:** Limit to 1-2 concurrent episodes to avoid memory exhaustion.

**Within-Episode Parallelism:**
- Bleep detection can run parallel to speaker ID (independent stages)
- Embedding generation parallelized across speaker clusters

#### 4. Streaming & Chunking

**For Long Episodes (>2 hours):**
```python
def process_long_episode(audio_path: str, chunk_duration: int = 1800):
    """Process in 30-minute chunks to reduce memory footprint."""
    chunks = split_audio_into_chunks(audio_path, chunk_duration)
    results = []

    for chunk in chunks:
        result = process_chunk(chunk)
        results.append(result)
        del result  # Free memory
        gc.collect()

    return merge_chunk_results(results)
```

#### 5. Database Optimization

**Indexes:**
```sql
CREATE INDEX idx_bleep_episode_time ON bleep_events(episode_id, start_ms);
CREATE INDEX idx_bleep_person ON bleep_events(person_id);
CREATE INDEX idx_embedding_speaker ON voice_embeddings(speaker_id);
CREATE INDEX idx_episode_speaker ON episode_speakers(episode_id, speaker_id);
```

**Query Optimization:**
```python
# Use prepared statements
cursor.execute("SELECT * FROM bleep_events WHERE episode_id = ? AND start_ms BETWEEN ? AND ?",
               (episode_id, start_ms, end_ms))

# Batch inserts
cursor.executemany("INSERT INTO voice_embeddings VALUES (?, ?, ?)", batch_data)
```

#### 6. Memory Management

**Explicit Cleanup:**
```python
import gc

def process_with_cleanup(audio_file: str):
    try:
        result = heavy_processing(audio_file)
        return result
    finally:
        gc.collect()  # Force garbage collection
        clear_model_cache()
```

**Memory Profiling:**
```python
import tracemalloc

tracemalloc.start()
# ... processing ...
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

### Performance Monitoring

**Key Metrics to Track:**
- Stage processing times
- Memory usage per stage
- Disk I/O patterns
- Cache hit rates
- API call latencies
- Database query times

**Logging:**
```python
@profile_time
def process_episode(episode_path: str):
    logger.info(f"Starting processing: {episode_path}")
    start_time = time.time()

    # ... processing ...

    duration = time.time() - start_time
    logger.info(f"Processing completed in {duration:.2f}s")
```

---

## Deployment Architecture

### Target Environment

**Platform:** macOS 12.0+ (Monterey, Ventura, Sonoma)
**Architecture:** Apple Silicon (M1/M2/M3) or Intel x86_64
**Dependencies:** Python 3.11, FFmpeg

### Deployment Options

#### Option 1: Python Package (pip install)

**Pros:**
- Easy updates via pip
- Standard Python tooling
- Developer-friendly

**Cons:**
- Requires Python installation
- Manual dependency management
- Not self-contained

**Installation:**
```bash
pip install show-scribe

# Verify installation
show-scribe --version

# Download models
show-scribe download-models
```

#### Option 2: Homebrew Formula (Planned)

**Pros:**
- Easy installation for end users
- Automatic dependency resolution
- macOS-native package manager

**Cons:**
- Requires Homebrew
- Larger initial download

**Installation:**
```bash
brew tap yourusername/show-scribe
brew install show-scribe
```

#### Option 3: Standalone .app Bundle (Future)

**Pros:**
- True macOS application
- Double-click to run
- Bundled Python runtime
- Pre-downloaded models

**Cons:**
- Large download size (~5 GB)
- Code signing requirements
- Notarization needed for macOS Gatekeeper

**Build Process:**
```bash
# Using py2app
python setup.py py2app

# Sign application
codesign --deep --force --verify --verbose --sign "Developer ID" ShowScribe.app

# Notarize for Gatekeeper
xcrun notarytool submit ShowScribe.dmg
```

### Deployment Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       User's Mac                            │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │               Applications Layer                      │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │  │
│  │  │ Terminal   │  │  Finder    │  │  Browser   │    │  │
│  │  │    .app    │  │    .app    │  │    .app    │    │  │
│  │  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘    │  │
│  └─────────┼────────────────┼────────────────┼──────────┘  │
│            │                │                │              │
│            ▼                │                ▼              │
│  ┌──────────────────┐      │      ┌──────────────────┐    │
│  │  show-scribe CLI │      │      │ Streamlit Server │    │
│  │  (Python package)│      │      │  (localhost:8501)│    │
│  └──────────┬───────┘      │      └──────────┬───────┘    │
│             │                │                │              │
│             └────────────────┼────────────────┘              │
│                              │                               │
│  ┌──────────────────────────┼────────────────────────────┐ │
│  │          File System                                    │ │
│  │  ~/Documents/VoiceTranscriptTool/                      │ │
│  │  ├── voice_bank/                                       │ │
│  │  │   ├── voice_bank.db                                 │ │
│  │  │   ├── embeddings/                                   │ │
│  │  │   └── audio_samples/                                │ │
│  │  └── shows/                                            │ │
│  │      └── [ShowName]/episodes/                          │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐  │
│  │           System Dependencies                        │  │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐   │  │
│  │  │FFmpeg  │  │Python3 │  │ PyTorch│  │ SQLite │   │  │
│  │  │ (brew) │  │ (system)│  │ (pip) │  │(system)│   │  │
│  │  └────────┘  └────────┘  └────────┘  └────────┘   │  │
│  └─────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Configuration Management

**Configuration Hierarchy:**
```
1. Default configuration (hardcoded in code)
2. Global configuration (~/.show-scribe/config.yaml)
3. Show configuration (~/Documents/.../shows/[ShowName]/show_config.json)
4. Environment variables (SHOW_SCRIBE_*)
5. Command-line arguments (highest priority)
```

**Configuration Loading:**
```python
def load_configuration() -> Config:
    config = load_defaults()
    config.update(load_global_config())
    config.update(load_show_config())
    config.update(load_env_vars())
    config.update(parse_cli_args())
    return config
```

### Update Strategy

**Versioning:** Semantic Versioning (MAJOR.MINOR.PATCH)

**Update Process:**
```bash
# Check for updates
show-scribe update --check

# Perform update
show-scribe update

# Or via pip
pip install --upgrade show-scribe
```

**Backward Compatibility:**
- Database schema migrations handled automatically
- Configuration file format versioned
- Voice bank format versioned with migration path

---

## Reliability & Availability

### Fault Tolerance

#### Checkpoint System

**Checkpoint Files:**
```
episodes/S01E01/checkpoints/
├── stage_1_complete.flag  # Audio extraction
├── stage_2_complete.flag  # Transcription
├── stage_3_complete.flag  # Diarization
├── stage_3b_complete.flag # Bleep detection
├── stage_4_complete.flag  # Speaker ID
└── stage_5_complete.flag  # Manual review
```

**Resume Logic:**
```python
def resume_processing(episode_id: str):
    checkpoints = load_checkpoints(episode_id)

    if checkpoints.stage_5_complete:
        logger.info("Episode already processed")
        return

    if not checkpoints.stage_1_complete:
        extract_audio(episode_id)
        save_checkpoint(episode_id, "stage_1")

    if not checkpoints.stage_2_complete:
        transcribe(episode_id)
        save_checkpoint(episode_id, "stage_2")

    # ... continue from last checkpoint
```

#### Error Recovery

**Retry Strategy:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def call_api_with_retry(endpoint: str, data: dict):
    response = requests.post(endpoint, json=data)
    response.raise_for_status()
    return response.json()
```

**Graceful Degradation:**
- If Whisper API fails → fallback to local Whisper
- If speaker ID fails → mark as unknown (user reviews later)
- If bleep detection fails → skip (optional feature)

#### Data Integrity

**Database Transactions:**
```python
def add_voice_embedding(speaker_id: int, embedding_data: np.ndarray):
    conn = get_db_connection()
    try:
        conn.execute("BEGIN TRANSACTION")

        # Insert metadata
        cursor = conn.execute(
            "INSERT INTO voice_embeddings (speaker_id, embedding_path) VALUES (?, ?)",
            (speaker_id, embedding_path)
        )
        embedding_id = cursor.lastrowid

        # Save embedding file
        np.save(embedding_path, embedding_data)

        conn.execute("COMMIT")
    except Exception as e:
        conn.execute("ROLLBACK")
        raise e
```

**File System Atomicity:**
```python
import tempfile
import shutil

def write_file_atomically(dest_path: str, content: str):
    # Write to temporary file first
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    # Atomic rename (works on most file systems)
    shutil.move(tmp_path, dest_path)
```

### Availability Targets

| Component | Target Availability | Acceptable Downtime |
|-----------|---------------------|---------------------|
| Core processing | 99% | 87 hours/year |
| Streamlit UI | 95% | 18 days/year (dev tool) |
| Database | 99.9% | 8.76 hours/year |
| Voice bank | 99.9% | 8.76 hours/year |

**Note:** As a single-user desktop application, traditional availability metrics are less critical. Focus is on fault tolerance and data integrity.

### Disaster Recovery

**Backup Strategy:**
```bash
# Automated backup (daily cron job)
show-scribe backup --output ~/backups/voice_bank_$(date +%Y%m%d).zip

# Backup includes:
# - voice_bank.db
# - embeddings/
# - audio_samples/
# - show_config.json files
```

**Restore Process:**
```bash
# Restore from backup
show-scribe restore --input ~/backups/voice_bank_20251016.zip

# Verify integrity
show-scribe voice-bank verify
```

**Recovery Time Objective (RTO):** <15 minutes
**Recovery Point Objective (RPO):** 24 hours (daily backups)

---

## Monitoring & Observability

### Logging Strategy

#### Log Levels

| Level | Usage | Examples |
|-------|-------|----------|
| **DEBUG** | Detailed diagnostic info | "Loaded 1024 audio samples", "Cosine similarity: 0.87" |
| **INFO** | Significant events | "Processing started", "Stage 2 complete" |
| **WARNING** | Potentially problematic | "Low confidence match", "API rate limit approaching" |
| **ERROR** | Error events | "Failed to extract audio", "Database query failed" |
| **CRITICAL** | System failure | "Database corruption detected", "Out of memory" |

#### Log Format

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('~/.show-scribe/logs/app.log'),
        logging.StreamHandler()  # Console output
    ]
)

# Structured logging
logger.info("Processing episode", extra={
    "episode_id": "TheOffice_S02E05",
    "stage": "transcription",
    "duration_seconds": 45.2,
    "memory_mb": 2048
})
```

#### Log Rotation

```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    '~/.show-scribe/logs/app.log',
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=5
)
```

### Metrics Collection

**Key Metrics:**

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ProcessingMetrics:
    episode_id: str
    stage: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    memory_peak_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: str = None
```

**Metrics Storage:**
```sql
CREATE TABLE processing_metrics (
    id INTEGER PRIMARY KEY,
    episode_id TEXT,
    stage TEXT,
    duration_seconds REAL,
    memory_peak_mb REAL,
    success BOOLEAN,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Metrics Reporting:**
```bash
# CLI command to view metrics
show-scribe metrics --episode TheOffice_S02E05

# Output:
Stage              Duration  Memory    Status
-----------------  --------  --------  ------
Audio Extraction   1.2s      512 MB    ✓
Transcription      45.3s     2048 MB   ✓
Diarization        32.1s     1536 MB   ✓
Speaker ID         0.8s      256 MB    ✓
```

### Health Checks

**System Health Check:**
```python
def system_health_check() -> dict:
    return {
        "ffmpeg_installed": check_ffmpeg(),
        "python_version": check_python_version(),
        "disk_space_gb": get_available_disk_space(),
        "memory_available_gb": get_available_memory(),
        "database_accessible": check_database(),
        "models_downloaded": check_models()
    }
```

**Periodic Health Checks:**
- Before each processing run
- On application startup
- On-demand via CLI command

### Error Tracking

**Error Context Capture:**
```python
def capture_error_context(exception: Exception, episode_id: str):
    context = {
        "error_type": type(exception).__name__,
        "error_message": str(exception),
        "episode_id": episode_id,
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "platform": platform.system(),
            "python_version": sys.version,
            "memory_available": psutil.virtual_memory().available
        },
        "stack_trace": traceback.format_exc()
    }

    # Log to file
    with open('~/.show-scribe/logs/errors.jsonl', 'a') as f:
        f.write(json.dumps(context) + '\n')
```

### Alerting (Future)

**Alert Conditions:**
- Processing failure rate >10%
- Disk space <5 GB
- Memory usage >90%
- API rate limit exceeded
- Database corruption detected

**Alert Channels:**
- Desktop notification (macOS Notification Center)
- Email (optional user configuration)
- Webhook (for automation)

---

## Scalability Roadmap

### Current Limitations (v1.0)

| Component | Limitation | Impact |
|-----------|----------|--------|
| **Single-threaded processing** | One episode at a time | Low throughput for batch processing |
| **In-memory voice bank** | All embeddings loaded | Memory limit ~1000 speakers |
| **Local file system** | No cloud storage | Limited collaboration |
| **SQLite database** | Single writer | No concurrent processing |
| **Manual review UI** | One user at a time | Bottleneck for large backlogs |

### Scalability Targets (v2.0)

| Metric | v1.0 | v2.0 Target |
|--------|------|-------------|
| Concurrent episodes | 1 | 5-10 |
| Voice bank size | 100 speakers | 1000+ speakers |
| Episodes per show | 500 | 5000+ |
| User concurrency | 1 | 10+ |
| Processing throughput | 1 episode/90 min | 5 episodes/90 min |

### Scaling Strategies

#### Horizontal Scaling

**Distributed Processing:**
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Worker 1    │     │  Worker 2    │     │  Worker 3    │
│  Episode A   │     │  Episode B   │     │  Episode C   │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       └────────────────────┼────────────────────┘
                            │
                    ┌───────▼────────┐
                    │  Job Queue     │
                    │  (Redis/RabbitMQ)│
                    └────────────────┘
```

**Implementation:**
```python
# Using Celery for distributed task processing
from celery import Celery

app = Celery('show_scribe', broker='redis://localhost:6379')

@app.task
def process_episode_async(episode_path: str):
    return process_episode(episode_path)

# Submit jobs
for episode in episodes:
    process_episode_async.delay(episode)
```

#### Vertical Scaling

**Database Upgrade:**
- Migrate from SQLite to PostgreSQL for concurrent writes
- Implement connection pooling
- Partition tables by show or date

**Storage Optimization:**
- Compress voice embeddings (quantization)
- Use sparse representation for embeddings
- Tiered storage (hot/warm/cold)

#### Caching & CDN

**Voice Bank Caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_speaker_embeddings(speaker_id: int) -> np.ndarray:
    return load_embeddings_from_disk(speaker_id)
```

**Model Caching:**
- Pre-load models on worker startup
- Share models across workers (memory-mapped files)

### Future Architecture (v3.0)

**Cloud-Native Design:**
```
┌────────────────────────────────────────────────────────────┐
│                      Cloud Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   S3/GCS     │  │  PostgreSQL  │  │  Redis Cache │    │
│  │  (Storage)   │  │  (Database)  │  │              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└──────────────────────────┬─────────────────────────────────┘
                           │
┌──────────────────────────┴─────────────────────────────────┐
│                    API Gateway (REST/GraphQL)               │
└──────────────────────────┬─────────────────────────────────┘
                           │
┌──────────────────────────┴─────────────────────────────────┐
│                  Processing Services                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  ASR     │  │ Diarize  │  │ Speaker  │  │ Analytics│  │
│  │ Service  │  │ Service  │  │    ID    │  │ Service  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Technology Decision Records

### TDR-001: Whisper for ASR

**Decision:** Use OpenAI Whisper (large-v3) as primary ASR engine

**Context:**
- Need highly accurate transcription
- Support for multiple languages
- Local processing capability essential

**Options Considered:**
1. Google Cloud Speech-to-Text
2. AWS Transcribe
3. AssemblyAI
4. OpenAI Whisper

**Decision Rationale:**
- **Accuracy:** Whisper achieves 3-5% WER on clear audio
- **Local Processing:** Can run entirely offline
- **Cost:** Open-source with free API option
- **Multilingual:** Supports 99 languages
- **Adoption:** Strong community support

**Consequences:**
- ✅ High accuracy and flexibility
- ✅ No vendor lock-in
- ⚠️  Slower than real-time locally
- ⚠️  Large model size (2.9 GB)

---

### TDR-002: Pyannote for Diarization

**Decision:** Use Pyannote.audio for speaker diarization

**Context:**
- Need speaker segmentation with high accuracy
- Must handle overlapping speech
- Prefer open-source solution

**Options Considered:**
1. Pyannote.audio
2. AssemblyAI (cloud)
3. Custom model
4. AWS Transcribe with diarization

**Decision Rationale:**
- **State-of-the-art:** Leading DER performance (10-15%)
- **Active Development:** Well-maintained project
- **Research Backing:** CNRS research lab
- **Extensible:** Can train custom models

**Consequences:**
- ✅ Excellent accuracy and flexibility
- ✅ Strong community and documentation
- ⚠️  Requires HuggingFace token (minor friction)
- ⚠️  Processing speed ~0.5-1x real-time

---

### TDR-003: SQLite for Voice Bank

**Decision:** Use SQLite as primary database

**Context:**
- Need persistent storage for voice profiles
- Single-user application (initially)
- Minimal setup requirements

**Options Considered:**
1. SQLite
2. PostgreSQL
3. MongoDB
4. JSON files

**Decision Rationale:**
- **Zero Configuration:** Bundled with Python
- **Reliable:** ACID-compliant, battle-tested
- **Portable:** Single file database
- **Performance:** Sufficient for v1.0 scale

**Consequences:**
- ✅ Simple deployment and backup
- ✅ No separate database server needed
- ⚠️  Single writer limitation (acceptable for v1.0)
- ⚠️  Migration path needed for multi-user (v2.0)

---

### TDR-004: Streamlit for Review UI

**Decision:** Use Streamlit for initial review interface

**Context:**
- Need quick prototype for user review workflow
- Developer team primarily Python-focused
- UI not customer-facing (internal tool)

**Options Considered:**
1. Streamlit
2. Flask + HTML/CSS/JS
3. Electron + React
4. Native SwiftUI

**Decision Rationale:**
- **Rapid Development:** Pure Python, no frontend knowledge needed
- **Good Enough:** Sufficient for review workflows
- **Iterations:** Easy to iterate and add features
- **Migration Path:** Can replace with native UI in v2.0

**Consequences:**
- ✅ Fast time-to-market
- ✅ Easy maintenance
- ⚠️  Limited UI customization
- ⚠️  Not suitable for distribution (needs server)

---

### TDR-005: DSP-Based Bleep Detection

**Decision:** Use signal processing (DSP) for bleep detection, not ML

**Context:**
- Need to detect tone/mute/noise censorship
- Prefer interpretable approach
- Limited training data available

**Options Considered:**
1. DSP feature extraction + rules
2. CNN classifier
3. Hybrid DSP + ML
4. Cloud API (AssemblyAI content moderation)

**Decision Rationale:**
- **Interpretability:** Clear logic for debugging
- **No Training Data Needed:** Rules-based approach
- **Fast Inference:** Real-time capable
- **Sufficient Accuracy:** >90% recall on synthetic tests

**Consequences:**
- ✅ Explainable results
- ✅ No model training overhead
- ⚠️  May require tuning per show
- ⚠️  ML enhancement possible in v2.0

---

## Architecture Evolution Plan

### Phase 1: MVP (v1.0) - Q4 2025 ✅ Current

**Focus:** Core functionality with manual oversight

**Deliverables:**
- ✅ Audio extraction
- ✅ Whisper transcription (local + API)
- ✅ Pyannote diarization
- ✅ Voice bank with speaker ID
- ✅ DSP-based bleep detection
- ✅ Streamlit review UI
- ✅ Checkpoint/resume system
- ✅ Speaking time analytics

**Architecture Characteristics:**
- Single-user desktop application
- Sequential pipeline processing
- SQLite database
- Local file storage
- Manual review for edge cases

---

### Phase 2: Automation & Scale (v1.5) - Q1 2026

**Focus:** Reduce manual intervention, handle larger workloads

**Features:**
- 🔄 Batch processing (multiple episodes in parallel)
- 🔄 Improved confidence calibration (fewer false positives)
- 🔄 Voice bank management UI enhancements
- 🔄 Automated folder monitoring
- 🔄 Cross-episode analytics dashboard
- 🔄 Enhanced bleep word suggestion (context-aware)

**Architecture Changes:**
- Multi-threaded episode processing (2-3 concurrent)
- Improved caching for voice bank queries
- Optimized database indexes
- Background worker for monitoring

---

### Phase 3: Collaboration & Cloud (v2.0) - Q2 2026

**Focus:** Multi-user support and cloud integration

**Features:**
- 📅 User authentication and roles
- 📅 Cloud storage integration (S3, Google Drive)
- 📅 Shared voice banks across team
- 📅 Real-time collaborative review
- 📅 REST API for integrations
- 📅 Webhook notifications

**Architecture Changes:**
- Migrate to PostgreSQL for concurrent access
- Add authentication layer (OAuth2)
- Implement API gateway
- Cloud-native deployment option (Docker, Kubernetes)
- WebSocket for real-time updates

---

### Phase 4: Intelligence & Integration (v3.0) - Q3 2026

**Focus:** Advanced AI features and ecosystem integration

**Features:**
- 📅 Emotion/sentiment detection
- 📅 Topic modeling and summarization
- 📅 Video editor plugins (Premiere Pro, Final Cut)
- 📅 Real-time streaming transcription
- 📅 Multi-language translation
- 📅 Advanced bleep detection (ML-enhanced)

**Architecture Changes:**
- Microservices architecture
- ML model serving layer (TensorFlow Serving)
- Event-driven pipeline (Kafka/RabbitMQ)
- Plugin SDK for third-party integrations
- GraphQL API

---

### Phase 5: Enterprise (v4.0) - Q4 2026+

**Focus:** Enterprise-grade features and compliance

**Features:**
- 📅 Role-based access control (RBAC)
- 📅 Audit trails and compliance reporting
- 📅 Voice biometric encryption
- 📅 Multi-tenancy support
- 📅 SLA-backed processing guarantees
- 📅 Advanced analytics and BI integration

**Architecture Changes:**
- Multi-region deployment
- High availability setup (load balancing, failover)
- Data residency options (GDPR compliance)
- Enterprise SSO integration (SAML, LDAP)
- Dedicated support infrastructure

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **ASR** | Automatic Speech Recognition - converting speech to text |
| **Diarization** | Speaker segmentation - determining "who spoke when" |
| **Embedding** | Mathematical vector representation of voice characteristics |
| **DER** | Diarization Error Rate - primary metric for diarization accuracy |
| **WER** | Word Error Rate - primary metric for transcription accuracy |
| **Voice Bank** | Database of speaker profiles and voice embeddings |
| **Checkpoint** | Saved state marker enabling resume from failures |
| **Bleep** | Censorship event (tone, mute, or noise mask) |
| **DSP** | Digital Signal Processing - analysis of audio signals |
| **STFT** | Short-Time Fourier Transform - spectral analysis technique |

### References

**Academic Papers:**
- Radford et al. (2023): "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper)
- Bredin et al. (2020): "pyannote.audio: neural building blocks for speaker diarization"
- Wan et al. (2018): "Generalized End-to-End Loss for Speaker Verification"

**Technical Documentation:**
- OpenAI Whisper: https://github.com/openai/whisper
- Pyannote.audio: https://github.com/pyannote/pyannote-audio
- FFmpeg: https://ffmpeg.org/documentation.html

**Industry Standards:**
- NIST Speaker Recognition Evaluation: https://www.nist.gov/itl/iad/mig/speaker-recognition
- SRT Subtitle Format: https://en.wikipedia.org/wiki/SubRip

### Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2025-10-10 | Architecture Team | Initial draft |
| 0.5 | 2025-10-13 | Architecture Team | Added bleep detection architecture |
| 1.0 | 2025-10-16 | Architecture Team | Complete architecture document |

---

**Document Status:** ✅ APPROVED FOR IMPLEMENTATION
**Next Review Date:** 2026-01-16
**Document Owner:** Technical Architecture Team
**Classification:** Internal Technical Documentation

---

*This architecture document is a living document and will be updated as the system evolves. All architectural decisions should be documented as TDRs and referenced in this document.*
