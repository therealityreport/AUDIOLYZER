# ASR & Diarization Evaluation Harness

A batch-ready evaluation framework for comparing ASR and speaker diarization tools on macOS.

## 🆕 What's New

**Recent Updates** (see [UPDATES.md](UPDATES.md) for details):

- ✅ **SpeechBrain Diarization**: Full implementation with ECAPA embeddings + spectral/agglomerative clustering
- ✅ **Model Downloader**: One-command sherpa-onnx model download & validation (`download_models.py`)
- ✅ **Robust Alignment**: Epsilon tolerance, tie resolution ("UNK" labels), strict SAD mode
- ✅ **Pinned Versions**: All dependencies pinned for reproducibility
- ✅ **Environment Logging**: Automatic capture of Python/package/git versions to `outputs/logs/env.json`

## Overview

This harness evaluates:
- **ASR Tools**: faster-whisper, sherpa-onnx
- **Diarization Tools**: pyannote.audio 3.x, SpeechBrain

It produces:
- Word-level transcriptions with timestamps
- Speaker diarization in RTTM format
- Word-to-speaker alignment
- Comprehensive metrics: WER, cpWER, DER, JER, RTF
- Summary CSV with all results

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install core dependencies
pip install -r requirements.txt

# Install optional tools for scoring
pip install dscore  # For DER calculation
```

### 2. Install Models

#### faster-whisper

Models download automatically on first use. To pre-download:

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3", compute_type="int8_float16")
```

#### sherpa-onnx

Download models manually:

```bash
# Example: Paraformer English
mkdir -p models/sherpa-onnx/paraformer
cd models/sherpa-onnx/paraformer

# Download from sherpa-onnx releases or model zoo
# Place model.onnx and tokens.txt in this directory
```

See: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html

#### pyannote.audio

Authenticate with Hugging Face:

```bash
pip install huggingface_hub[cli]
huggingface-cli login
```

Accept model terms at:
- https://huggingface.co/pyannote/speaker-diarization@2.1
- https://huggingface.co/pyannote/segmentation-3.0

#### SpeechBrain

Models download automatically. Note: The SpeechBrain diarization wrapper is a placeholder and needs to be implemented based on your specific recipe.

### 3. Prepare Data

```bash
# Place audio/video files in data/clips/
mkdir -p data/clips

# Copy your test clips (MP4, MOV, WAV, etc.)
cp /path/to/your/clips/* data/clips/

# Optionally, add reference transcripts and RTTM files
# for scoring (same basename as clips)
mkdir -p data/refs/transcripts
mkdir -p data/refs/rttm
```

### 4. Configure

Edit [`eval.yaml`](eval.yaml) to:
- Enable/disable tools
- Set model paths
- Configure tool parameters
- Specify clips to process (or leave empty to process all)

### 5. Run

```bash
# Run all stages
python eval/runner.py --config eval/eval.yaml

# Or run specific stages
python eval/runner.py --stage 1  # Extract WAVs
python eval/runner.py --stage 2  # Run ASR
python eval/runner.py --stage 3  # Run diarization
python eval/runner.py --stage 4  # Align words to speakers
python eval/runner.py --stage 5  # Score and generate summary

# Verbose output
python eval/runner.py --verbose
```

## Pipeline Stages

### Stage 1: Extract WAVs

Converts input media (MP4/MOV/etc.) to mono 16kHz WAV using ffmpeg.

**Output**: `outputs/wavs/*.wav`

### Stage 2: Run ASR

Runs enabled ASR tools on each WAV file.

**Output**: `outputs/asr/{tool}/*.json`

**Schema**:
```json
{
  "file": "clip.wav",
  "language": "en",
  "segments": [
    {
      "start": 12.34,
      "end": 15.67,
      "text": "Hello world",
      "words": [
        {"w": "Hello", "start": 12.34, "end": 12.60, "p": 0.92},
        {"w": "world", "start": 12.61, "end": 13.01, "p": 0.95}
      ]
    }
  ]
}
```

### Stage 3: Run Diarization

Runs enabled diarization tools on each WAV file.

**Output**: `outputs/dia/{tool}/*.rttm`

**Format**: Standard RTTM
```
SPEAKER clip 1 12.34 3.45 <NA> <NA> SPEAKER_01 <NA> <NA>
```

### Stage 4: Align

Aligns ASR words to diarization speakers for each tool combination.

**Output**: `outputs/aligned/*.jsonl`

**Schema** (one JSON object per line):
```json
{"file":"clip","tool_asr":"faster_whisper","tool_dia":"pyannote","w":"Hello","start":12.34,"end":12.60,"speaker":"SPEAKER_01","prob":0.92}
```

### Stage 5: Score

Computes metrics and generates summary CSV.

**Outputs**:
- `outputs/scores/summary.csv`: Main results table
- `outputs/logs/runtime.jsonl`: Detailed timing logs

**Metrics**:
- **WER**: Word Error Rate (if reference transcripts provided)
- **cpWER**: Conversation-preserving WER (if speaker-segmented references provided)
- **DER**: Diarization Error Rate (if reference RTTM provided)
- **JER**: Jaccard Error Rate
- **RTF**: Real-Time Factor (processing_time / audio_duration)

## Configuration

Key settings in [`eval.yaml`](eval.yaml):

```yaml
paths:
  input_dir: "data/clips"
  output_dir: "outputs"

audio:
  sample_rate: 16000
  channels: 1

asr:
  faster_whisper:
    enabled: true
    model: "large-v3"
    compute_type: "int8_float16"
    vad_filter: true

  sherpa_onnx:
    enabled: false  # Set to true when models are downloaded
    model_path: "models/sherpa-onnx/paraformer"

diarization:
  pyannote:
    enabled: true
    pipeline: "pyannote/speaker-diarization@2.1"

  speechbrain:
    enabled: false  # Placeholder implementation

references:
  transcripts_dir: null  # Set to "data/refs/transcripts" if available
  rttm_dir: null         # Set to "data/refs/rttm" if available
```

## Output Structure

```
outputs/
├── wavs/              # Extracted mono 16kHz WAVs
├── asr/
│   ├── faster_whisper/
│   │   └── clip.json
│   └── sherpa_onnx/
│       └── clip.json
├── dia/
│   ├── pyannote/
│   │   └── clip.rttm
│   └── speechbrain/
│       └── clip.rttm
├── aligned/
│   └── clip_faster_whisper_pyannote.jsonl
├── scores/
│   └── summary.csv
└── logs/
    └── runtime.jsonl
```

## Summary CSV Format

```csv
clip,asr,dia,duration,wer,cpwer,der,jer,rtf_asr,rtf_dia,notes
clip1,faster_whisper,pyannote,120.5,12.3,N/A,8.5,0.12,0.15,0.85,
clip2,sherpa_onnx,pyannote,95.2,15.7,N/A,9.1,0.15,0.08,0.92,
```

## Standalone Tool Usage

### Run ASR only

```python
from eval.asr import FasterWhisperASR

asr = FasterWhisperASR(model="large-v3", compute_type="int8_float16")
result = asr.transcribe("audio.wav")
```

### Run diarization only

```python
from eval.diar import PyAnnoteDiarization

diarizer = PyAnnoteDiarization()
segments = diarizer.diarize("audio.wav")  # Returns [(start, end, speaker), ...]
```

### Align words to speakers

```bash
python eval/align/align_words_to_rttm.py \
  --asr outputs/asr/faster_whisper/clip.json \
  --rttm outputs/dia/pyannote/clip.rttm \
  --out aligned.jsonl \
  --asr-tool faster_whisper \
  --dia-tool pyannote
```

## Advanced CLI Usage

### Tool Pair Selection

Run specific combinations of ASR and diarization tools:

```bash
# Run only faster-whisper + pyannote
python eval/runner.py --pairs "asr=faster_whisper dia=pyannote"

# Run multiple combinations
python eval/runner.py --pairs "asr=faster_whisper,sherpa_onnx dia=pyannote,speechbrain"
```

### Skip Stages

Skip specific stages when re-running:

```bash
# Skip extraction and scoring (useful for testing alignment changes)
python eval/runner.py --skip extract,score

# Skip by number
python eval/runner.py --skip 1,5

# Available stages: extract (1), asr (2), dia (3), align (4), score (5)
```

### Cleanup

Clean old outputs before running:

```bash
# Remove outputs/tmp and logs older than 7 days
python eval/runner.py --clean

# Can combine with other flags
python eval/runner.py --clean --skip score
```

### Version Information

The runner automatically prints a version banner showing:
- Harness version and git commit
- Python and platform info
- Installed tool versions (faster-whisper, pyannote, etc.)

```bash
python eval/runner.py --help
```

## Automation & CI

### GitHub Actions

The harness includes a smoke test workflow that runs on every push:

**Workflow**: [`.github/workflows/smoke.yml`](.github/workflows/smoke.yml)

**What it tests**:
1. Installs dependencies on macOS-latest
2. Generates a 30-second test audio file
3. Runs stages 1-4 (skips scoring)
4. Verifies outputs exist:
   - `outputs/asr/faster_whisper/sample.json`
   - `outputs/dia/pyannote/sample.rttm`
   - `outputs/aligned/sample_faster_whisper_pyannote.jsonl`
   - `outputs/logs/env.json`
5. Uploads artifacts for inspection

**Run locally**:

```bash
# Generate test sample
python tests/generate_sample.py

# Copy to data/clips
cp tests/data/sample.wav data/clips/

# Run smoke test
python eval/runner.py --skip score --verbose
```

**Badge**:

```markdown
![Smoke Test](https://github.com/YOUR_ORG/YOUR_REPO/workflows/Smoke%20Test/badge.svg)
```

### Manual Testing

```bash
# Test installation
python eval/test_install.py

# Download models
python eval/utils/download_models.py --sherpa paraformer-en

# Generate sample audio
python tests/generate_sample.py

# Run on sample
cp tests/data/sample.wav data/clips/
python eval/runner.py --config eval/eval.yaml
```

### Continuous Evaluation

For ongoing benchmarking:

```bash
# Run weekly with cleanup
0 0 * * 0 cd /path/to/eval && python runner.py --clean --config eval.yaml

# Or use --pairs to test specific configurations
python runner.py --pairs "asr=faster_whisper dia=pyannote" --clean
```

## Diarization DER Benchmark

Benchmark the shipping presets against curated reality-TV clips:

```bash
# Evaluate default presets defined in bench_der.DEFAULT_CONFIGS
python eval/diar/bench_der.py

# Override manifest or configs
python eval/diar/bench_der.py \\
  --manifest eval/diar/manifest.json \\
  --configs configs/dev.yaml configs/prod.yaml configs/reality_tv.yaml
```

- `eval/diar/manifest.json` lists each clip, the audio asset to run, and RTTM/segment references.
- Results land in `eval/diar/results/der_results.csv` with per-clip DER, JER, detected speaker count, and overlap ratios.
- `eval/diar/results/README.md` is regenerated on every run with per-config averages plus any clip-level outliers (`DER > mean + 0.05`).
- Hugging Face auth (`PYANNOTE_TOKEN`, etc.) must be available because the harness instantiates the production diarizer stack.

## Requirements

- **macOS** (tested on Apple Silicon)
- **Python** 3.9+
- **ffmpeg** (for audio extraction)
  ```bash
  brew install ffmpeg
  ```
- **PyTorch** (installed with dependencies)

## Troubleshooting

### faster-whisper fails to load

- Ensure you have sufficient RAM (large-v3 needs ~10GB)
- Try `medium.en` model instead
- Check Apple Silicon compatibility

### pyannote authentication errors

```bash
huggingface-cli login
# Then accept terms on model pages
```

### sherpa-onnx models not found

- Download models from sherpa-onnx model zoo
- Update `model_path` in config
- Verify `tokens.txt` and `model.onnx` are present

### dscore not available

DER/JER will fall back to simple Python implementation if dscore is not installed. For more accurate results:

```bash
pip install dscore
```

## Performance Tips

1. **Use VAD filtering** (enabled by default in faster-whisper) to skip silent regions
2. **Process clips in parallel** by running multiple instances with different clip sets
3. **Cache embeddings** (pyannote does this automatically)
4. **Use quantized models** (int8_float16) on Apple Silicon for faster inference

## Decision Matrix

The harness evaluates tools across:

| Criterion | Weight | Metrics |
|-----------|--------|---------|
| Accuracy | 50% | WER (25%), cpWER (10%), DER (15%) |
| Runtime | 20% | ASR RTF (10%), DIA RTF (10%) |
| Stability | 10% | Variance in repeated runs |
| Simplicity | 10% | Installation, setup complexity |
| Integration | 10% | API quality, output formats |

## Citation

If you use this harness in research, please cite the underlying tools:

- **faster-whisper**: https://github.com/guillaumekln/faster-whisper
- **sherpa-onnx**: https://github.com/k2-fsa/sherpa-onnx
- **pyannote.audio**: https://github.com/pyannote/pyannote-audio
- **SpeechBrain**: https://github.com/speechbrain/speechbrain

## License

MIT License - see individual tool licenses for their respective terms.

## Contributing

Contributions welcome! Areas for improvement:

1. Full SpeechBrain diarization implementation
2. Additional ASR tools (Wav2Vec2, NeMo, etc.)
3. Advanced cpWER calculation with speaker mapping
4. GPU support for faster-whisper
5. Batch processing optimizations

## Contact

For issues or questions, please file an issue in the repository.
