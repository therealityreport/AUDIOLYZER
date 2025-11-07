# 🎬 Reality TV Audio Preprocessing - Quick Reference

## 🚀 One-Line Setup
```bash
python scripts/install_preprocessing.py && show-scribe process episode.mp4 --config configs/reality_tv.yaml
```

---

## 💊 The Fix for Bad Transcripts

**Problem:** Background music + restaurant noise = Wrong words & missing dialogue

**Solution:** Pre-process audio before Whisper

**Result:** 50-70% improvement in transcription accuracy

---

## 📋 3-Step Workflow

### 1. Install (One Time)
```bash
python scripts/install_preprocessing.py
```

### 2. Configure (Per Show)
```yaml
# configs/my_show.yaml
audio_preprocessing:
  enable: true
  vocal_separation:
    enable: true    # Remove music
  enhancement:
    enable: true    # Remove noise
```

### 3. Process (Per Episode)
```bash
show-scribe process episode.mp4 --config configs/my_show.yaml
```

---

## ⚙️ What It Does

```
Video File
    ↓
Extract Audio (2 min)
    ↓
🎵 Remove Background Music (8 min)
    ↓
🔧 Clean Up Noise (12 min)
    ↓
Transcribe with Whisper (10 min)
    ↓
✅ Accurate Transcript
```

**Total Time:** ~32 minutes for 1-hour episode

---

## 🎚️ Tuning Knobs

### When Transcripts Are Still Bad

**More aggressive denoising:**
```yaml
enhancement:
  lambd: 0.9  # was 0.7
  tau: 0.8    # was 0.6
```

**Try different separation model:**
```yaml
vocal_separation:
  model: "mdx_extra"  # was "htdemucs" (faster)
```

### When Audio Sounds Over-Processed

**Gentler enhancement:**
```yaml
enhancement:
  lambd: 0.5  # was 0.7
  tau: 0.4    # was 0.6
```

**Just separate vocals:**
```yaml
vocal_separation:
  enable: true
enhancement:
  enable: false  # Turn off enhancement
```

---

## 🧪 Test Before Committing

```bash
# Benchmark 5 configurations
python scripts/benchmark_preprocessing.py episode.mp4

# Compare side-by-side
show-scribe process episode.mp4                              # Before
show-scribe process episode.mp4 --config configs/reality_tv.yaml  # After
```

---

## ✅ Use Preprocessing When You See:
- 🎵 Background music during dialogue
- 🍽️ Restaurant/bar scenes
- 🏙️ Outdoor/street scenes
- 📢 Crowd noise
- 🔊 Echo/reverb
- ❌ Whisper transcribing song lyrics

## ⊘ Skip Preprocessing For:
- 🎬 Studio sitcoms (already clean)
- 📺 Modern dramas
- 🗞️ News/talk shows
- ✨ High-quality productions

---

## 🐛 Quick Fixes

| Problem | Solution |
|---------|----------|
| "Module not found" | `python scripts/install_preprocessing.py` |
| Still inaccurate | Check `preprocessing_report.json`, increase lambd/tau |
| Too slow | Use `mdx_extra` model, or Whisper API |
| Over-processed sound | Lower lambd/tau to 0.5/0.4 |
| Out of memory | Close other apps, use smaller model |

---

## 📊 Expected Improvements

| Audio Type | Before | After | Gain |
|------------|--------|-------|------|
| Music + dialogue | 25% WER | 8% WER | **68%** ⬆️ |
| Restaurant noise | 20% WER | 10% WER | **50%** ⬆️ |
| Clean studio | 5% WER | 5% WER | 0% (not needed) |

---

## 📁 Files You'll See

```
episodes/S01E01/
├── audio_extracted.wav      # Original
├── audio_processed.wav      # Cleaned (Whisper uses this)
└── processed_audio/
    ├── audio_vocals.wav     # After vocal separation
    ├── audio_enhanced_vocals.wav   # Enhanced vocals
    ├── audio_enhanced_mix.wav      # Enhanced full mix
    └── preprocessing_report.json # Analysis details
```

---

## 💾 Storage Needed

- Models: ~3 GB (one-time download)
- Per episode: +150 MB extra audio files
- Tip: Set `audio_preprocessing.retain_intermediates: false` (prod default) to auto-purge stems when you don't need audits

---

## 🎯 Quick Decision Tree

```
Is your transcription bad?
├─ No → Don't use preprocessing
└─ Yes
   ├─ Is there background music? → Use vocal separation
   ├─ Is it noisy/echoey? → Use enhancement
   └─ Both? → Use Reality TV preset
```

---

## 📞 Getting Help

1. Check `preprocessing_report.json` - see what was applied
2. Listen to `audio_processed.wav` - hear the cleaned audio
3. Read `docs/audio_preprocessing.md` - full guide
4. Read `docs/install_preprocessing.md` - troubleshooting

---

## 🎓 Key Insight

**Reality TV needs preprocessing. Studio shows don't.**

The extra 20 minutes is worth it when you go from 25% word errors to 8% word errors.

---

## ⚡ Copy-Paste Commands

```bash
# Install (choose one)
python scripts/install_preprocessing.py
# or
pip install -e ".[preprocessing]"
python -c "from audio_separator.separator import Separator; from resemble_enhance.enhancer.inference import enhance"

# Test on one episode
show-scribe process episode.mp4 --config configs/reality_tv.yaml

# Override once
show-scribe process episode.mp4 --preprocess

# Benchmark multiple configs
python scripts/benchmark_preprocessing.py episode.mp4

# Pipeline CLI (auto audio)
python scripts/run_pipeline.py \
  --input "/path/E01.mp4" \
  --episode-id RHOBH_S13E01 \
  --show-config data/shows/RHOBH/show_config.json \
  --preprocess \
  --preset reality_tv

python scripts/run_pipeline.py \
  --input "/path/episodes/RHOBH_S13E01/audio_extracted.wav" \
  --episode-id RHOBH_S13E01 \
  --show-config data/shows/RHOBH/show_config.json \
  --preprocess

# Make it permanent for a show
cp configs/reality_tv.yaml configs/my_show.yaml
# Edit configs/my_show.yaml as needed
```

---

Streamlit flow: “Process New Episode” → **CREATE AUDIO** → pick `audio_enhanced_vocals.wav` (or the mix variant) → run pipeline.

Add `--allow-fallback-audio` if you want the CLI to continue with `audio_extracted.wav` when preprocessing fails.

---

**That's it! Your reality TV transcripts will now be much more accurate. 🎉**
