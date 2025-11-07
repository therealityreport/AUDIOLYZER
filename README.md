# Show-Scribe

**Automated Voice Identification & Transcription Tool for TV Show Episodes**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![macOS](https://img.shields.io/badge/macOS-12.0+-000000.svg?style=flat&logo=apple&logoColor=F0F0F0)](https://www.apple.com/macos/)

> Transform hours of TV episodes into speaker-labeled transcripts with censorship detection in minutes.

---

## 🎯 Overview

Show-Scribe is a macOS-based tool that automatically:
- **Transcribes** TV show episodes with high accuracy (Whisper)
- **Identifies speakers** using persistent voice recognition
- **Detects censorship** (bleeps, mutes, noise masks) and attributes them to speakers
- **Generates analytics** on speaking time and redacted content
- **Exports** formatted transcripts in multiple formats (TXT, SRT, JSON, CSV)

### Key Features

✨ **Automatic Speaker Identification**
- Builds a "voice bank" that learns cast member voices over time
- Auto-labels speakers with confidence scores (high/medium/low)
- Handles similar voices, guest stars, and voice effects

🔍 **Censorship Detection**
- Detects bleeps (tones), mutes, and noise masks
- Attributes each censorship event to the speaking cast member
- Exports reviewable table: `WORD | PERSON | TIMESTAMP | SENTENCE`

📊 **Speaking Time Analytics**
- Calculate duration, word count, and segment count per speaker
- Generate visual charts and reports
- Track censorship patterns across episodes

🔄 **Resume Capability**
- Checkpoint system saves progress at each stage
- Resume interrupted processing without starting over
- Graceful error recovery

💾 **Persistent Voice Bank**
- SQLite database stores speaker profiles and voice embeddings
- Grows more accurate with each processed episode
- Export/import for backup and team collaboration

---

## 🚀 Quick Start

### Prerequisites

- macOS 12.0 (Monterey) or later
- Python 3.11+
- 16 GB RAM recommended
- 50 GB free storage

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/show-scribe.git
cd show-scribe

# Run setup (installs dependencies, FFmpeg, downloads models)
make setup

# Verify installation
show-scribe --version
```

### Basic Usage

```bash
# Process a single episode
show-scribe process path/to/episode.mp4

# Override audio preprocessing from the command line
show-scribe process path/to/episode.mp4 --preprocess   # force on
show-scribe process path/to/episode.mp4 --no-preprocess  # force off

# Resume interrupted processing
show-scribe resume TheOffice_S02E05

# Launch review UI
show-scribe ui

# Export voice bank
show-scribe voice-bank export backup.zip
```

---

## 🔊 Audio Preprocessing

- `audio_preprocessing.enable` is disabled in the base `configs/dev.yaml` and `configs/prod.yaml`; enable it per show by layering a preset such as `configs/reality_tv.yaml` or by passing `--preprocess/--no-preprocess` on the CLI.
- Install the optional preprocessing stack with either `python scripts/install_preprocessing.py` (auto-verifies imports) or `pip install -e ".[preprocessing]" && python -c "from audio_separator.separator import Separator; from resemble_enhance.enhancer.inference import enhance"`.
- Switch between `resemble` (Demucs + resemble-enhance) and `clearervoice` (ClearerVoice Studio) by setting `audio_preprocessing.enhancement.provider`; the UI exposes this as a dropdown in the Audio Preprocessing panel. ClearerVoice requires `pip install clearvoice` and emits both enhanced vocals and an enhanced mix.
- When enabled, the pipeline writes `processed_audio/audio_vocals.wav`, `processed_audio/audio_enhanced_vocals.wav`, `processed_audio/audio_enhanced_mix.wav`, `processed_audio/audio_processed.wav`, and `processed_audio/preprocessing_report.json`; `audio_processed.wav` is always the ASR/diarization input.
- Control whether stems stay on disk with `audio_preprocessing.retain_intermediates` (defaults to `true` locally, `false` in production exports).
- Benchmark impact before rollout with `python scripts/benchmark_preprocessing.py /path/to/noisy_episode.mp4`; append the resulting WER deltas JSON to your rollout notes.

### CLI Pipeline Examples

```bash
# Video → auto-extract + enhance, then run
python scripts/run_pipeline.py \
  --input "/path/E01.mp4" \
  --episode-id RHOBH_S13E01 \
  --show-config data/shows/RHOBH/show_config.json \
  --preprocess \
  --preset reality_tv

# Audio already extracted → enhance only, then run
python scripts/run_pipeline.py \
  --input "/path/episodes/RHOBH_S13E01/audio_extracted.wav" \
  --episode-id RHOBH_S13E01 \
  --show-config data/shows/RHOBH/show_config.json \
  --preprocess
```

Streamlit flow: “Process New Episode” → **CREATE AUDIO** → pick `audio_enhanced_vocals.wav` (or `audio_enhanced_mix.wav`) → run pipeline.

Add `--allow-fallback-audio` if you want the CLI to continue with `audio_extracted.wav` when preprocessing fails.

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [PRD.md](./PRD.md) | Product Requirements Document - Complete specification |
| [DIRECTORY_STRUCTURE.md](./DIRECTORY_STRUCTURE.md) | Repository layout and organization |
| [TECH_SPEC.md](./TECH_SPEC.md) | Technical specifications and dev stack |
| [Full PRD (Google Doc)](https://docs.google.com/document/d/16F9ritPvmaU5qDXzKdpILwv9NmyqIk5amhyiisX7QRQ/edit) | Detailed requirements document |

---

## 🎬 Workflow

```
1. Drop video file → 2. Extract audio → 3. Transcribe (Whisper)
         ↓                    ↓                    ↓
   4. Diarize (Pyannote) → 5. Detect bleeps → 6. Identify speakers
         ↓                    ↓                    ↓
   7. Review unknowns → 8. Generate transcript → 9. Export analytics
```

### Processing Stages

| Stage | Duration | Output |
|-------|----------|--------|
| Audio Extraction | ~2 min | WAV file (16kHz mono) |
| Transcription (API) | ~10 min | Time-stamped text |
| Transcription (Local) | ~45 min | Time-stamped text |
| Diarization | ~45 min | Speaker segments |
| Bleep Detection | ~3 min | Censorship events |
| Speaker Identification | <1 min | Speaker labels |
| Manual Review | ~5 min | Confirmed labels |
| **Total (API)** | **~30 min** | **Complete outputs** |
| **Total (Local)** | **~2 hours** | **Complete outputs** |

---

## 📁 Data Organization

```
~/Documents/VoiceTranscriptTool/
├── voice_bank/
│   ├── voice_bank.db          # Speaker profiles & embeddings
│   ├── embeddings/            # Voice fingerprints (.npy)
│   └── audio_samples/         # Reference clips
├── shows/
│   └── TheOffice/
│       ├── show_config.json
│       └── episodes/
│           └── S02E05/
│               ├── transcript_final.txt
│               ├── transcript_final.srt
│               ├── analytics.json
│               ├── bleeps.csv
│               └── checkpoints/
├── config/
│   └── app_settings.json
└── backups/
```

---

## 🎯 Use Cases

### Content Review
Track censored language across episodes to ensure consistent editorial standards.

### Analytics
Analyze speaking time distribution among cast members to ensure balanced screen presence.

### Accessibility
Generate accurate speaker-labeled subtitles for improved accessibility.

### Research
Create searchable transcripts for TV show analysis, dialogue studies, or content research.

### Compliance
Document redacted content for regulatory or network compliance purposes.

---

## 🔧 Configuration

### Show Settings (`show_config.json`)

```json
{
  "show_name": "The Office",
  "default_speakers": ["Michael Scott", "Jim Halpert"],
  "confidence_threshold": 0.80,
  "bleeps": {
    "enable": true,
    "detect_types": ["tone", "mute", "noise"],
    "suggest_words": true
  }
}
```

### App Settings (`app_settings.json`)

```json
{
  "whisper": {
    "use_local": true,
    "model": "large-v3"
  },
  "voice_bank": {
    "max_embeddings_per_speaker": 20,
    "auto_backup": true
  }
}
```

---

## 📊 Output Files

Each processed episode generates:

| File | Format | Content |
|------|--------|---------|
| `transcript_final.txt` | Plain text | Speaker-labeled dialogue |
| `transcript_final.srt` | SRT | Subtitle format |
| `transcript_final.json` | JSON | Full metadata |
| `analytics.json` | JSON | Speaking time & stats |
| `bleeps.csv` | CSV | Censorship table |
| `bleeps.json` | JSON | Detailed bleep data |

### Example Transcript Output

```
Episode: The Office - S02E05
Duration: 22:03
Processed: October 16, 2025

---

[00:00:15] Michael Scott: Good morning everyone!

[00:00:22] Jim Halpert: Hey Michael, did you see the memo?

[00:00:28] Michael Scott: What the [BLEEP] is going on?
```

Need the richer alignment annotations (speaker confidence, per-word spans)? Use:

```python
from src.show_scribe.pipelines.transcript.export_text import render_plain_text_with_alignment
from src.show_scribe.pipelines.transcript.export_srt import render_srt_with_alignment

detailed_text = render_plain_text_with_alignment(document)
detailed_srt = render_srt_with_alignment(document)
```

### Example Bleeps CSV

```csv
WORD,PERSON,TIMESTAMP,SENTENCE USED IN
fuck,Michael Scott,00:00:28.420,"What the [BLEEP] is going on?"
shit,Dwight Schrute,00:12:15.380,"This is [BLEEP] ridiculous."
```

---

## 🛠️ Development

### Setup Development Environment

```bash
# Install dev dependencies
make dev

# Run linting
make lint

# Run type checking
make typecheck

# Run tests
make test

# REQUIRED: Sync cast directories after updating show_config files
make sync-cast       # or: python scripts/maintenance/sync_cast_configs.py --dry-run
```

### Project Structure

```
show-scribe/
├── src/show_scribe/       # Core application
│   ├── pipelines/         # Processing stages
│   ├── storage/           # Database & file management
│   └── utils/             # Shared utilities
├── ui/                    # Streamlit review interface
├── tests/                 # Test suite
├── scripts/               # Maintenance scripts
└── docs/                  # Documentation
```

### Running Tests

```bash
# All tests
make test

# Unit tests only
make test-unit

# Integration tests
make test-integration

# Edge case tests
pytest tests/edge_cases/
```

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`make test lint`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## 🎓 Technology Stack

**Core Technologies:**
- **Transcription:** OpenAI Whisper (large-v3)
- **Diarization:** Pyannote.audio
- **Voice Recognition:** Resemblyzer / Pyannote embeddings
- **Bleep Detection:** NumPy/SciPy DSP processing
- **Database:** SQLite
- **UI:** Streamlit

**See [TECH_SPEC.md](./TECH_SPEC.md) for complete technical details.**

---

## 📈 Performance

### Accuracy Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Word Error Rate | <10% | Whisper typically achieves 3-5% |
| Diarization Error Rate | ~10-15% | Pyannote baseline |
| Speaker ID Accuracy | >90% | After 5+ episodes processed |
| Bleep Detection Recall | ≥90% | On intentional censorship |
| Bleep Detection Precision | ≥90% | <10% false positives |

### Processing Speed

- **With Cloud API:** ~30 minutes for 1-hour episode
- **Local Processing:** ~2 hours for 1-hour episode
- **Apple Silicon (M2/M3):** ~50% faster than Intel

---

## 🔒 Privacy & Data

- **Local Processing:** All data stays on your machine when using local models
- **Voice Bank:** Speaker voice prints stored securely in SQLite database
- **No Telemetry:** No usage data sent to external servers
- **Backup Control:** Full control over voice bank exports and backups

---

## 🐛 Troubleshooting

### Common Issues

**Issue:** `FFmpeg not found`
```bash
# Install FFmpeg via Homebrew
brew install ffmpeg
```

**Issue:** `Out of memory during processing`
```bash
# Use chunking for large episodes
show-scribe process --chunk-size 30 episode.mp4
```

**Issue:** `Whisper model download fails`
```bash
# Manually download models
python scripts/setup/download_models.py
```

**Issue:** `Unknown speakers not saving`
```bash
# Check voice bank database
show-scribe voice-bank verify
```

For more troubleshooting, see [docs/troubleshooting.md](./docs/troubleshooting.md)

---

## 📝 Roadmap

### v1.0 (MVP) - Current
- ✅ Audio extraction & transcription
- ✅ Speaker diarization
- ✅ Voice bank with speaker identification
- ✅ Bleep detection & attribution
- ✅ Basic Streamlit UI
- ✅ Speaking time analytics

### v1.1 - Planned
- 🔄 Automated folder monitoring
- 🔄 Batch processing multiple episodes
- 🔄 Enhanced voice bank management UI
- 🔄 Cross-episode analytics dashboard

### v2.0 - Future
- 📅 Cloud storage integration (S3, GCS)
- 📅 Real-time processing
- 📅 Video editor integration
- 📅 Emotion/tone detection
- 📅 Collaborative voice banks

---

## 🙏 Acknowledgments

Built with:
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [Pyannote.audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) - Voice embeddings
- [FFmpeg](https://ffmpeg.org/) - Media processing
- [Streamlit](https://streamlit.io/) - UI framework

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/show-scribe/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/show-scribe/discussions)
- **Email:** support@show-scribe.com

---

## ⭐ Star History

If you find Show-Scribe useful, please consider starring the repository!

---

**Made with ❤️ for content creators, producers, and TV show enthusiasts**

**Version:** 1.0.0 | **Last Updated:** October 16, 2025
