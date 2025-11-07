# Quick Start Guide

Get the evaluation harness running in 10 minutes.

## Step 1: Install Dependencies (2 min)

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install faster-whisper pyannote.audio pyyaml numpy torch

# Install ffmpeg (if not already installed)
brew install ffmpeg

# Authenticate with Hugging Face for pyannote
pip install huggingface_hub[cli]
huggingface-cli login
```

Accept model terms:
- https://huggingface.co/pyannote/speaker-diarization@2.1
- https://huggingface.co/pyannote/segmentation-3.0

## Step 2: Add Test Clips (1 min)

```bash
# Copy 2-3 audio/video files to data/clips/
cp ~/path/to/interview.mp4 data/clips/
cp ~/path/to/dialog.wav data/clips/
```

## Step 3: Configure (30 sec)

Edit `eval.yaml`:

```yaml
asr:
  faster_whisper:
    enabled: true
    model: "medium.en"  # Start with smaller model

  sherpa_onnx:
    enabled: false      # Skip for now (needs model download)

diarization:
  pyannote:
    enabled: true

  speechbrain:
    enabled: false      # Not implemented yet
```

## Step 4: Run (5 min)

```bash
# Run full pipeline
python eval/runner.py --config eval/eval.yaml --verbose
```

Watch the progress through 5 stages:
1. Extract WAVs
2. Run faster-whisper ASR
3. Run pyannote diarization
4. Align words to speakers
5. Generate summary

## Step 5: Check Results (1 min)

```bash
# View summary CSV
cat outputs/scores/summary.csv

# Or open in Excel/Numbers
open outputs/scores/summary.csv
```

## What You'll See

```csv
clip,asr,dia,duration,wer,cpwer,der,jer,rtf_asr,rtf_dia,notes
interview,faster_whisper,pyannote,120.5,N/A,N/A,N/A,N/A,0.15,0.85,
dialog,faster_whisper,pyannote,95.2,N/A,N/A,N/A,N/A,0.12,0.92,
```

WER/DER show as `N/A` without reference transcripts - that's normal!

## Next Steps

### Add Ground Truth (Optional)

For WER/DER scoring:

```bash
# Create reference transcripts
mkdir -p data/refs/transcripts
echo "This is the reference transcript" > data/refs/transcripts/interview.txt

# Create reference RTTM
mkdir -p data/refs/rttm
cat > data/refs/rttm/interview.rttm << 'EOF'
SPEAKER interview 1 0.00 5.23 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER interview 1 5.23 8.45 <NA> <NA> SPEAKER_01 <NA> <NA>
EOF

# Update eval.yaml
# references:
#   transcripts_dir: "data/refs/transcripts"
#   rttm_dir: "data/refs/rttm"
```

### Try Larger Models

```yaml
asr:
  faster_whisper:
    model: "large-v3"  # Better accuracy, slower
```

### Add sherpa-onnx

Easy way with model downloader:

```bash
# Download Paraformer English model
python eval/utils/download_models.py --sherpa paraformer-en

# Enable in config
# asr:
#   sherpa_onnx:
#     enabled: true
```

Or manual:
1. Download models from https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
2. Place in `models/sherpa-onnx/paraformer/`
3. Enable in config

## Advanced Usage

### Run Specific Tools

Run only certain tool combinations:

```bash
# Only faster-whisper + pyannote
python eval/runner.py --pairs "asr=faster_whisper dia=pyannote"

# Test all combinations
python eval/runner.py --pairs "asr=faster_whisper,sherpa_onnx dia=pyannote,speechbrain"
```

### Skip Stages

Save time by skipping stages:

```bash
# Skip scoring (useful during development)
python eval/runner.py --skip score

# Skip extraction (WAVs already exist)
python eval/runner.py --skip extract

# Skip multiple stages
python eval/runner.py --skip extract,score
```

### Clean Old Outputs

```bash
# Remove tmp files and logs older than 7 days
python eval/runner.py --clean

# Combine with other options
python eval/runner.py --clean --skip score
```

### Quick Smoke Test

```bash
# Generate test audio
python tests/generate_sample.py

# Copy to data/clips
cp tests/data/sample.wav data/clips/

# Run without scoring
python eval/runner.py --skip score --verbose
```

## Troubleshooting

### Out of memory

Use smaller model:
```yaml
model: "base.en"  # or "tiny.en"
```

### pyannote fails

Check authentication:
```bash
huggingface-cli whoami
```

### No output files

Check logs:
```bash
python eval/runner.py --verbose
```

## Test Clip Suggestions

For comprehensive testing, include:

1. **Solo interview** - One clear speaker
2. **Dialog** - Two people, clean audio
3. **Party scene** - Multiple speakers + music
4. **Phone call** - Compressed audio
5. **Voiceover** - VO + background audio
6. **Cross-talk** - Overlapping speakers
7. **Noisy** - Background noise
8. **Off-mic** - Distant/poor quality

Each clip: 1-3 minutes for fast iteration.

## Performance Expectations

On Apple Silicon M1/M2:

| Model | RTF (faster-whisper) | Accuracy |
|-------|---------------------|----------|
| tiny.en | 0.05 | Low |
| base.en | 0.08 | Medium |
| medium.en | 0.12 | Good |
| large-v3 | 0.20 | Excellent |

RTF < 1.0 = faster than real-time

pyannote RTF: typically 0.5-1.5 depending on audio length and speaker count.

## Need Help?

See full [README.md](README.md) for:
- Detailed configuration options
- Advanced usage
- API documentation
- Troubleshooting guide
