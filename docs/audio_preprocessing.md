# Audio Preprocessing for Reality TV

**Problem:** Reality TV shows often have poor transcription accuracy due to:
- Background music during dialogue
- Restaurant/crowd ambient noise
- Poor microphone placement
- Reverb and echo from large spaces

**Solution:** Pre-process audio before transcription using vocal separation and enhancement.

---

## Quick Start

### 1. Install Dependencies

```bash
python scripts/install_preprocessing.py
# or
pip install -e ".[preprocessing]"
python -c "from audio_separator.separator import Separator; from resemble_enhance.enhancer.inference import enhance"
```

Both flows install and verify:
- `audio-separator` - Separates vocals from background music/noise
- `resemble-enhance` - Enhances audio quality with denoising
- `clearvoice` - ClearerVoice Studio stack (optional, quality mode)

### 2. Use Reality TV Preset

```bash
show-scribe process episode.mp4 --config configs/reality_tv.yaml
```

Need a quick override?

```bash
# Force-enable preprocessing for a single run
show-scribe process episode.mp4 --preprocess

# Force-disable (even if the config enables it)
show-scribe process episode.mp4 --no-preprocess
```

---

## What It Does

### Before Preprocessing
```
🎵🎵🎵 [music] "I can't believe..." [dishes clanking] "...this happened" [crowd chatter] 🎵🎵🎵
```
**Whisper hears:** Music + Speech + Noise → Confused transcription

### After Preprocessing

**Step 1: Vocal Separation** (removes background music)
```
🗣️ "I can't believe this happened" 🗣️
```

**Step 2: Enhancement** (removes noise, reduces reverb)
```
🗣️✨ "I can't believe this happened" ✨🗣️
```
**Whisper hears:** Clean speech → Accurate transcription

---

## Configuration Options

### Default (Global Config)
```yaml
audio_preprocessing:
  enable: false
  retain_intermediates: true
```

### Automatic Mode (Scoped)
```yaml
audio_preprocessing:
  enable: "auto"
  retain_intermediates: true
  vocal_separation:
    enable: "auto"  # Detects music automatically
  enhancement:
    enable: "auto"  # Detects noise automatically
```

### Always On (Reality TV)
```yaml
audio_preprocessing:
  enable: true
  retain_intermediates: true
  vocal_separation:
    enable: true  # Always separate
    model: "htdemucs"
  enhancement:
    enable: true  # Always enhance
    lambd: 0.7  # Aggressive denoising
    tau: 0.6    # Moderate dereverberation
```

### Disabled (Clean Studio Audio)
```yaml
audio_preprocessing:
  enable: false
```

---

## Vocal Separation Models

| Model | Quality | Speed | Best For |
|-------|---------|-------|----------|
| `htdemucs` | ⭐⭐⭐⭐⭐ | Slower | Reality TV, music-heavy |
| `mdx_extra` | ⭐⭐⭐⭐ | Faster | Quick processing |
| `demucs` | ⭐⭐⭐ | Fast | Basic separation |

**Reality TV Recommendation:** `htdemucs` (best quality)

---

## Enhancement Settings

### Provider (`provider`)

- `resemble` *(default)* — Demucs → `resemble-enhance`; fastest and fully offline.
- `clearervoice` — Uses ClearerVoice Studio (`pip install clearvoice`) to run separation/enhancement in one stack. Produces both `audio_enhanced_vocals.wav` and `audio_enhanced_mix.wav`, and is recommended for "quality" runs or particularly noisy scenes.

Choose the provider in config (`audio_preprocessing.enhancement.provider`) or through the Streamlit UI dropdown. When ClearerVoice is selected you can further tune models under `audio_preprocessing.enhancement.clearervoice` (override separation/enhancement model names or target sample rate).

### Denoising Strength (`lambd`)
- `0.5` - Gentle (preserve naturalness)
- `0.7` - **Moderate (Reality TV default)**
- `0.9` - Aggressive (very noisy audio)

### Dereverberation Strength (`tau`)
- `0.4` - Light (small rooms)
- `0.6` - **Moderate (Reality TV default)**
- `0.8` - Heavy (large venues)

### Quality vs Speed (`nfe`)
- `32` - Fast (good quality)
- `64` - **Balanced (Reality TV default)**
- `128` - Slow (best quality)

---

## Outputs & Contract

- `processed_audio/audio_vocals.wav` — vocals stem (retained unless `retain_intermediates: false`).
- `processed_audio/audio_enhanced_vocals.wav` — denoised vocals (retained unless purged).
- `processed_audio/audio_enhanced_mix.wav` — denoised full mix (retained unless purged).
- `processed_audio/audio_processed.wav` — final cleaned audio within the processed folder.
- `processed_audio/preprocessing_report.json` — structured metadata emitted for every run.
- `audio_processed.wav` — canonical episode artifact used by ASR/diarization and referenced in orchestrator metadata alongside the report path.

> When `retain_intermediates` is `false`, stems are removed after the final copy and the report records the purge status.

---

## Performance Impact

### Processing Time (1-hour episode)

| Configuration | Extraction | Preprocessing | Transcription | Total |
|---------------|------------|---------------|---------------|-------|
| **No preprocessing** | 2 min | 0 min | 10 min | **12 min** |
| **Vocal separation** | 2 min | +8 min | 10 min | **20 min** |
| **Enhancement** | 2 min | +15 min | 10 min | **27 min** |
| **Both (Reality TV)** | 2 min | +20 min | 10 min | **32 min** |

**Trade-off:** +20 minutes processing = Significantly better transcription

---

## Benchmark Workflow

1. Run `python scripts/benchmark_preprocessing.py /path/to/noisy_episode.mp4` on a representative scene.
2. Review the emitted JSON (WER deltas, processing timings) and store it alongside rollout notes (e.g., `reports/benchmarks/<show>.json`).
3. Use the values to tune `enhancement.lambd`/`enhancement.tau` with the [Quick Reference](./QUICK_REFERENCE.md) before enabling preprocessing for the show.

---

## When to Use What

### ✅ Use Preprocessing For:
- Reality TV shows
- Restaurant/bar scenes
- Outdoor locations
- Music competition shows
- Older shows (poor audio quality)
- Field recordings

### ⊘ Skip Preprocessing For:
- Studio sitcoms (The Office, Friends)
- News/talk shows
- Modern scripted dramas
- High-quality professional productions

> Rollout policy: keep `audio_preprocessing.enable` off globally, benchmark noisy shows with `python scripts/benchmark_preprocessing.py /path/to/noisy_episode.mp4`, then enable via a preset (e.g., `configs/reality_tv.yaml`) once WER gains are recorded.

---

## Benchmarking

Test preprocessing on your show:

```bash
python scripts/benchmark_preprocessing.py episode.mp4 \
  --reference-transcript data/reference/rhobh_s05e01.txt \
  --output-json reports/rhobh_s05e01_preproc_benchmark.json

# Compare ClearerVoice vs current stack on a clip
python scripts/bench_clearervoice_vs_current.py episode.wav \
  --reference-transcript data/reference/rhobh_s05e01.txt \
  --results-file results/bench_clearervoice.json
```

`benchmark_preprocessing.py` runs 5 tests:
1. Baseline (no preprocessing)
2. Vocal separation only
3. Enhancement only
4. Both (Reality TV preset)
5. Aggressive enhancement

The script prints a comparison table (latency + WER deltas) and stores detailed
results, including the preprocessed audio paths. Use `--scenarios` to run a
subset (e.g. `--scenarios baseline both`) when time is tight. Pair it with
`bench_clearervoice_vs_current.py` to capture a focused comparison between the
legacy stack and ClearerVoice; that script writes runtime, file-size, and
optional WER deltas to `results/bench_clearervoice.json` for regression tracking.

---

## Generated Artifacts

When preprocessing is enabled, each episode produces:

| Artifact | Path | Description |
|----------|------|-------------|
| `audio_processed.wav` | `data/shows/<show>/episodes/<episode>/…` | Final audio used for ASR/diarization |
| `audio_vocals.wav` | `processed_audio/audio_vocals.wav` | Demucs-separated vocals stem |
| `audio_enhanced_vocals.wav` | `processed_audio/audio_enhanced_vocals.wav` | Primary enhanced track |
| `audio_enhanced_mix.wav` | `processed_audio/audio_enhanced_mix.wav` | Enhanced mix (ambience preserved) |
| `audio_processed.wav` | `processed_audio/audio_processed.wav` | Copy of the final audio within the processed folder |
| `preprocessing_report.json` | `processed_audio/preprocessing_report.json` | Metrics (SNR, music ratio), timings, applied steps |

If preprocessing is disabled, the pipeline still writes `processed_audio/audio_processed.wav`
so downstream tooling always has a consistent location for the final media asset.

---

## Troubleshooting

### "audio-separator not installed"
```bash
pip install audio-separator
```

### "resemble-enhance not installed"
```bash
pip install resemble-enhance
```

### Preprocessing too slow
- Use `mdx_extra` model instead of `htdemucs`
- Lower `nfe` to 32
- Use API transcription instead of local

### Audio sounds over-processed
- Lower `lambd` to 0.5
- Lower `tau` to 0.4
- Try vocal separation only (disable enhancement)

### No improvement in accuracy
- Your audio may already be clean enough
- Check if music is actually present during dialogue
- Verify Whisper is using `large-v3` model

---

## Advanced: Custom Presets

Create a custom preset for your show:

```yaml
# configs/my_show.yaml
audio_preprocessing:
  enable: true

  vocal_separation:
    enable: true
    model: "htdemucs"

  enhancement:
    enable: true
    nfe: 64
    lambd: 0.6  # Adjust for your audio
    tau: 0.5    # Adjust for your audio

whisper:
  model: "large-v3"
  beam_size: 10
  initial_prompt: "Custom context about your show..."
```

---

## Technical Details

### Vocal Separation (Demucs)
Uses hybrid transformer-demucs to separate:
- Vocals (speech)
- Bass
- Drums
- Other (instruments, ambient)

We extract the **vocals** stem and discard the rest.

### Enhancement (Resemble-Enhance)
Two-stage process:
1. **Denoising:** Removes background noise using diffusion model
2. **Dereverberation:** Reduces echo/reverb using time-domain processing

### Audio Quality Analysis
Automatic detection analyzes:
- **SNR (Signal-to-Noise Ratio):** <15dB = noisy
- **Music Ratio:** >30% = music present
- **Speech Clarity:** <0.6 = unclear
- **Reverb Score:** >0.5 = echoey

Preprocessing is auto-enabled if any threshold exceeded.

---

## ✅ Merge Gate Checklist

- Python 3.11 is enforced locally (`.python-version`, Makefile) and in CI snippets.
- Audio artifacts match the contract (`processed_audio/audio_vocals.wav`, `processed_audio/audio_enhanced_vocals.wav`, `processed_audio/audio_enhanced_mix.wav`, `processed_audio/audio_processed.wav`, `processed_audio/preprocessing_report.json`; `audio_processed.wav` feeds ASR/diarization).
- Config defaults keep preprocessing off; CLI `--preprocess/--no-preprocess` overrides work and `retain_intermediates` toggles stem retention.
- Unit tests `tests/unit/pipelines/test_audio_preprocessing.py` and `tests/unit/pipelines/test_extract_audio.py` pass on Python 3.11.
- Benchmark JSON from `scripts/benchmark_preprocessing.py` shows a WER improvement on at least one noisy episode and is archived with rollout notes.
- CHANGELOG records the feature under the "Unreleased" section.

---

## Files Created

After preprocessing, each episode will have:

```
episodes/S01E01/
├── audio_extracted.wav              # Original from video
├── audio_processed.wav              # Final input to Whisper/diarization
└── processed_audio/
    ├── audio_vocals.wav             # After vocal separation
    ├── audio_enhanced_vocals.wav    # Enhanced vocals
    ├── audio_enhanced_mix.wav       # Enhanced full mix
    ├── audio_processed.wav          # Local copy of final audio
    └── preprocessing_report.json    # Quality analysis + timings
```

---

## Next Steps

1. ✅ Install dependencies
2. ✅ Process test episode with Reality TV preset
3. ✅ Compare transcription quality
4. ✅ Adjust settings if needed
5. ✅ Enable for all reality TV shows

**Questions?** Check the main README or open an issue.
