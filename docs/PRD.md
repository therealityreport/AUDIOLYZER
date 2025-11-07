# **PRD: Voice Identification & Transcript Tool for TV Show Episodes**

**Version:** 2.0 (Revised)
**Date:** October 16, 2025
**Project:** Show-Scribe Audio Analyzer

---

## **Table of Contents**

1. [Overview](#overview)
2. [Data Architecture](#data-architecture)
3. [Database Schema](#database-schema)
4. [Processing Pipeline](#processing-pipeline)
5. [Bleep Detection](#bleep-detection)
6. [File Outputs](#file-outputs)
7. [Configuration](#configuration)
8. [Technology Stack](#technology-stack)

---

## **Overview**

**Show-Scribe** is a macOS-based tool for automatically transcribing TV show episodes with speaker identification and censorship detection.

**Core Features:**
- Automatic speech-to-text transcription (Whisper)
- Speaker diarization and identification (Pyannote)
- Persistent voice bank that learns over time
- Bleep/censor detection and attribution
- Speaking time analytics
- Resume capability with checkpoints

**Full PRD:** https://docs.google.com/document/d/16F9ritPvmaU5qDXzKdpILwv9NmyqIk5amhyiisX7QRQ/edit

---

## **Data Architecture**

### **Directory Structure**

```
~/Documents/VoiceTranscriptTool/
├── voice_bank/
│   ├── voice_bank.db              # SQLite database
│   ├── embeddings/                # Voice embedding .npy files
│   └── audio_samples/             # Reference audio clips
├── shows/
│   └── [Show Name]/
│       ├── show_config.json
│       └── episodes/
│           └── S01E01/
│               ├── metadata.json
│               ├── audio_extracted.wav
│               ├── transcript_raw.json
│               ├── diarization.json
│               ├── transcript_final.txt
│               ├── transcript_final.srt
│               ├── transcript_final.json
│               ├── analytics.json
│               ├── bleeps.csv
│               ├── bleeps.json
│               └── checkpoints/
├── config/
│   └── app_settings.json
└── backups/
    └── voice_bank_YYYYMMDD_HHMMSS.db
```

### **File Naming Convention**

Format: `[ShowName]_S[Season]E[Episode]_[filetype].[ext]`

Examples:
- `TheOffice_S02E05_transcript_final.txt`
- `TheOffice_S02E05_bleeps.csv`
- `TheOffice_S02E05_analytics.json`

---

## **Database Schema**

### **Core Tables**

```sql
-- Speaker profiles
CREATE TABLE speaker_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    first_appearance_episode TEXT,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_segments INTEGER DEFAULT 0,
    notes TEXT
);

-- Voice embeddings (multiple per speaker)
CREATE TABLE voice_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id INTEGER NOT NULL,
    embedding_path TEXT NOT NULL,
    source_episode TEXT,
    source_timestamp TEXT,
    audio_sample_path TEXT,
    quality_score REAL,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (speaker_id) REFERENCES speaker_profiles(id) ON DELETE CASCADE
);

-- Processing history
CREATE TABLE processing_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT UNIQUE NOT NULL,
    episode_path TEXT NOT NULL,
    status TEXT NOT NULL,
    processing_started TIMESTAMP,
    processing_completed TIMESTAMP,
    total_speakers INTEGER,
    auto_identified_speakers INTEGER,
    manual_labeled_speakers INTEGER,
    transcription_confidence REAL,
    diarization_error_rate REAL,
    error_message TEXT
);

-- Episode-speaker relationships
CREATE TABLE episode_speakers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL,
    speaker_id INTEGER NOT NULL,
    total_duration_seconds REAL,
    total_word_count INTEGER,
    segment_count INTEGER,
    FOREIGN KEY (speaker_id) REFERENCES speaker_profiles(id) ON DELETE CASCADE,
    FOREIGN KEY (episode_id) REFERENCES processing_history(episode_id) ON DELETE CASCADE,
    UNIQUE(episode_id, speaker_id)
);

-- Bleep/censor events
CREATE TABLE bleep_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('tone','mute','noise')),
    confidence REAL NOT NULL,
    speaker_cluster_id TEXT,
    person_id INTEGER,
    sentence_text TEXT,
    word_label TEXT DEFAULT '[BLEEP]',
    word_label_source TEXT CHECK (word_label_source IN ('manual','inferred','unknown')) DEFAULT 'unknown',
    word_confidence REAL,
    audio_snippet_path TEXT,
    sfx_profile_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES speaker_profiles(id) ON DELETE SET NULL,
    FOREIGN KEY (sfx_profile_id) REFERENCES sfx_profiles(id) ON DELETE SET NULL
);

CREATE INDEX idx_bleep_episode_time ON bleep_events(episode_id, start_ms);
CREATE INDEX idx_bleep_person ON bleep_events(person_id);

-- SFX profiles for recurring beep signatures
CREATE TABLE sfx_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('tone','mute','noise')),
    center_freq_hz REAL,
    bandwidth_hz REAL,
    avg_duration_ms REAL,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## **Processing Pipeline**

### **Pipeline Stages**

1. **Audio Extraction** → `stage_1_complete.flag`
   - FFmpeg: Video → WAV (16kHz mono, 16-bit PCM, -20dB LUFS)
   - Output: `audio_extracted.wav`

2. **Transcription** → `stage_2_complete.flag`
   - Whisper (local or API): Audio → Text with timestamps
   - Output: `transcript_raw.json`

3. **Diarization** → `stage_3_complete.flag`
   - Pyannote: Detect speaker segments
   - Output: `diarization.json`

4. **Bleep Detection** → `stage_3b_complete.flag`
   - DSP analysis: Detect tone/mute/noise events
   - Align with diarization, attribute to speakers
   - Output: `bleeps_detected.json`

5. **Speaker Identification** → `stage_4_complete.flag`
   - Generate embeddings, match against voice bank
   - Confidence-based assignment (high/medium/low)
   - Insert [BLEEP] tokens in transcript

6. **Manual Labeling** → `stage_5_complete.flag`
   - User reviews unknown speakers
   - User reviews/edits bleep WORD labels
   - Voice bank updates

7. **Transcript Generation** → Final outputs
   - Generate .txt, .srt, .json formats
   - Export bleeps.csv and bleeps.json

8. **Analytics** → `analytics.json`
   - Speaking time per person
   - Bleep statistics

### **Confidence Thresholds**

- **High Confidence:** ≥0.85 → Auto-label (green ✓)
- **Medium Confidence:** 0.70-0.85 → Auto-label + flag for review (yellow ⚠)
- **Low Confidence:** <0.70 → Require manual labeling (red ✗)

---

## **Bleep Detection**

### **Detection Types**

| Type | Description | Duration | Characteristics |
|------|-------------|----------|-----------------|
| `tone` | Narrowband steady tone | 80-1000ms | 0.8-2.5 kHz, high crest factor |
| `mute` | Hard silence drop | Variable | ≥20-30dB drop during speech |
| `noise` | Broadband burst | 80-1000ms | High spectral flatness |

### **Detection Method**

1. **DSP Pipeline:**
   - STFT analysis on mono WAV
   - Extract: RMS, spectral centroid, bandwidth, crest, flatness
   - Apply detection rules per type
   - Merge events <120ms apart
   - Drop events <80ms (unless merged)

2. **Attribution:**
   - Align with diarization segments
   - Assign to speaker with max overlap
   - Split events at speaker boundaries

3. **Word Handling:**
   - Default: `[BLEEP]`
   - Optional: Context-based suggestion (inferred)
   - User confirmation → manual label

### **Bleep Table Output (CSV)**

```csv
WORD,PERSON,TIMESTAMP,SENTENCE USED IN
fuck,Alice Johnson,00:12:15.420,"What the [BLEEP] is going on?"
shit,Bob Smith,00:27:03.118,"This is some [BLEEP] right here."
[BLEEP],Alice Johnson,00:45:22.890,"I can't believe this [BLEEP] happened."
```

### **Configuration**

```json
{
  "bleeps": {
    "enable": true,
    "detect_types": ["tone", "mute", "noise"],
    "tone_min_dur_ms": 80,
    "merge_gap_ms": 120,
    "min_snr_db": 10,
    "suggest_words": true,
    "use_ml_classifier": false,
    "confidence_threshold": 0.70
  }
}
```

---

## **File Outputs**

### **Per Episode**

| File | Format | Content |
|------|--------|---------|
| `transcript_final.txt` | Plain text | Speaker-labeled dialogue with timestamps |
| `transcript_final.srt` | SRT | Subtitle format with speaker names |
| `transcript_final.json` | JSON | Full metadata + segments |
| `analytics.json` | JSON | Speaking time + bleep stats |
| `bleeps.csv` | CSV | WORD, PERSON, TIMESTAMP, SENTENCE |
| `bleeps.json` | JSON | Detailed bleep events |
| `metadata.json` | JSON | Processing info + checkpoints |

### **Transcript Format Example**

```
Episode: The Office - S02E05
Duration: 22:03
Processed: October 16, 2025

---

[00:00:15] Michael Scott: Good morning everyone!

[00:00:22] Jim Halpert: Hey Michael, did you see the memo?

[00:00:28] Michael Scott: Memo? I don't read memos.

[00:00:35] Pam Beesly: The memo is on your desk, Michael.
```

### **Analytics JSON Structure**

```json
{
  "episode_id": "TheOffice_S02E05",
  "total_dialogue_duration": 945,
  "speakers": [
    {
      "name": "Michael Scott",
      "duration_seconds": 315.5,
      "percentage_of_dialogue": 33.4,
      "word_count": 1847,
      "segment_count": 42
    }
  ],
  "bleeps": {
    "total_count": 18,
    "by_type": {"tone": 12, "mute": 5, "noise": 1},
    "by_person": {"Alice Johnson": 7, "Bob Smith": 5},
    "rate_per_minute": 0.81
  }
}
```

---

## **Configuration**

### **Show Config** (`show_config.json`)

```json
{
  "show_name": "The Office",
  "default_speakers": ["Michael Scott", "Jim Halpert", "Pam Beesly"],
  "confidence_threshold": 0.80,
  "primary_language": "en",
  "processing_preferences": {
    "use_local_whisper": true,
    "diarization_sensitivity": "medium",
    "minimum_segment_duration": 2.0
  },
  "bleeps": {
    "enable": true,
    "detect_types": ["tone", "mute", "noise"],
    "tone_min_dur_ms": 80,
    "merge_gap_ms": 120,
    "min_snr_db": 10,
    "suggest_words": true
  }
}
```

### **App Settings** (`app_settings.json`)

```json
{
  "whisper": {
    "use_local": true,
    "model": "large-v3",
    "api_key": null
  },
  "pyannote": {
    "model": "pyannote/speaker-diarization",
    "min_speakers": null,
    "max_speakers": null
  },
  "voice_bank": {
    "max_embeddings_per_speaker": 20,
    "auto_backup": true,
    "backup_frequency_days": 7
  },
  "ui": {
    "audio_player_preroll_ms": 2000,
    "waveform_enabled": true,
    "keyboard_shortcuts_enabled": true
  }
}
```

---

## **Technology Stack**

### **Core Components**

| Component | Technology | Purpose |
|-----------|------------|---------|
| Audio Extraction | FFmpeg | Video → WAV conversion |
| Transcription | OpenAI Whisper (large-v3) | Speech-to-text |
| Diarization | Pyannote.audio | Speaker segmentation |
| Voice Embeddings | Pyannote/Resemblyzer | Speaker identification |
| Bleep Detection | NumPy/SciPy (DSP) | Censorship detection |
| Database | SQLite | Voice bank storage |
| UI | Streamlit / SwiftUI | Review interface |

### **Python Libraries**

```
whisper (openai-whisper)
pyannote.audio
resemblyzer
ffmpeg-python
numpy
scipy
librosa (audio analysis)
scikit-learn (cosine similarity)
sqlite3 (built-in)
streamlit (UI)
pandas (data handling)
```

### **System Requirements**

**Minimum:**
- macOS 12.0 (Monterey)
- 8 GB RAM
- 10 GB storage
- Apple Silicon (M1+) or Intel i5 (2015+)

**Recommended:**
- macOS 14.0 (Sonoma)
- 16 GB RAM
- 50 GB storage
- Apple Silicon M2/M3

### **Performance Targets**

| Stage | Target Time (1-hour episode) |
|-------|------------------------------|
| Audio Extraction | <2 minutes |
| Transcription (API) | 6-12 minutes |
| Transcription (Local) | 30-60 minutes |
| Diarization | 30-60 minutes |
| Bleep Detection | <5 minutes |
| Speaker ID | <1 minute |
| **Total (API)** | **<30 minutes** |
| **Total (Local)** | **<2 hours** |

### **Accuracy Targets**

- Word Error Rate (WER): <10% (Whisper: 3-5%)
- Diarization Error Rate (DER): ~10-15%
- Speaker ID Accuracy: >90% (after 5+ episodes)
- Bleep Detection Recall: ≥90%
- Bleep Detection Precision: ≥90%

---

## **Workflow Summary**

```
1. User drops video file
2. Tool checks for existing checkpoints
3. Stage 1: Extract audio (FFmpeg)
4. Stage 2: Transcribe (Whisper)
5. Stage 3: Diarize (Pyannote)
6. Stage 3b: Detect bleeps (DSP)
7. Stage 4: Identify speakers (Voice bank matching)
8. Stage 5: User reviews unknowns + bleeps
9. Stage 6: Generate transcripts
10. Stage 7: Calculate analytics
11. Export all files + update voice bank
```

---

## **Edge Cases Handled**

✅ Background noise & music
✅ Voice effects (phone calls, distortion)
✅ Similar voices (twins, family)
✅ Guest speakers (single appearance)
✅ Off-screen narration
✅ Overlapping speech
✅ Audio quality variations
✅ Emotional variations (shouting, whispering)
✅ Accents and vocal changes over time

---

## **Name Management System**

### **Purpose**
Ensure consistent and correct spelling of cast member names across all outputs.

### **Implementation**

**1. Show Configuration**
```json
{
  "cast_members": [
    {
      "canonical_name": "Michael Scott",
      "common_misspellings": ["Micheal Scott", "Michael Scot"],
      "aliases": ["Michael", "Mike"]
    }
  ]
}
```

**2. Voice Bank Display Names**
- Database stores `display_name` field with canonical spelling
- All outputs reference this canonical form

**3. Auto-Correction Engine**
- Detects common misspellings
- Maps aliases to canonical names
- Fuzzy matching with confidence scores
- Logs all corrections

**4. UI Validation**
- Dropdowns with pre-populated cast names
- No manual typing of names (prevents errors)
- Auto-complete with suggestions

**5. Post-Processing**
- Final validation pass on all transcripts
- Ensures consistency across all exports

### **Benefits**
✅ Zero name spelling errors in outputs
✅ User defines names once in config
✅ Automatic correction of variations
✅ Consistent across TXT, SRT, JSON, CSV formats

**See:** NAME_MANAGEMENT.md for complete specification

---

## **Future Enhancements**

- Automated folder monitoring
- Cloud storage integration (S3, GCS)
- Real-time processing
- Multi-episode aggregation dashboard
- Video editor integration
- Emotion/tone detection
- Collaborative voice bank sharing
- Advanced interaction analytics

---

## **Key Documents**

- **Full PRD:** [Google Doc](https://docs.google.com/document/d/16F9ritPvmaU5qDXzKdpILwv9NmyqIk5amhyiisX7QRQ/edit)
- **Repo Structure:** See `DIRECTORY_STRUCTURE.md`
- **Architecture:** See `docs/architecture.md`
- **API Docs:** See `docs/api.md`
- **User Guide:** See `docs/user_guide.md`

---

**Document Version:** 2.0
**Last Updated:** October 16, 2025
**Author:** Product Team
**Status:** Ready for Implementation
