# Evaluation Harness - Recent Updates

## Summary of Improvements

This document describes the hardening and feature additions made to the ASR+Diarization evaluation harness.

## 1. ✅ SpeechBrain Diarization (COMPLETED)

**Status**: Fully implemented

**Implementation**: [eval/diar/speechbrain.py](eval/diar/speechbrain.py)

### Features:
- Full ECAPA-TDNN embedding extraction
- Sliding window approach (configurable window_size and hop_size)
- Two clustering methods:
  - Spectral clustering
  - Agglomerative clustering
- Automatic speaker count detection via silhouette score
- Oracle mode (if speaker count is known)
- Proper RTTM output with runtime tracking

### Configuration (eval.yaml):
```yaml
diarization:
  speechbrain:
    enabled: true
    source: "speechbrain/spkrec-ecapa-voxceleb"
    oracle_n_speakers: false
    num_speakers: null           # set if oracle_n_speakers=true
    cluster_method: "spectral"   # or "agglomerative"
    threshold: 0.5
    window_size: 1.5            # seconds
    hop_size: 0.75              # seconds
```

### Usage:
```python
from eval.diar import SpeechBrainDiarization

diarizer = SpeechBrainDiarization(
    cluster_method="spectral",
    window_size=1.5,
    hop_size=0.75
)
segments = diarizer.diarize("audio.wav")
```

## 2. ✅ Sherpa-ONNX Model Downloader (COMPLETED)

**Status**: Fully implemented

**Implementation**: [eval/utils/download_models.py](eval/utils/download_models.py)

### Features:
- One-command model download for sherpa-onnx
- Automatic extraction and validation
- Support for:
  - Paraformer EN/ZH
  - Zipformer EN/ZH
- Validates required files (tokens.txt, model.onnx, etc.)
- Provides actionable error messages if files missing

### Usage:
```bash
# Download Paraformer English model
python eval/utils/download_models.py --sherpa paraformer-en

# List available models
python eval/utils/download_models.py --list

# Check if model is installed
python eval/utils/download_models.py --check paraformer-en

# Custom output directory
python eval/utils/download_models.py --sherpa zipformer-en --output-dir custom/path
```

### Auto-validation:
The sherpa_onnx wrapper now validates model files at startup and provides helpful error messages:

```
FileNotFoundError: Model directory not found: models/sherpa-onnx/paraformer
Download with: python eval/utils/download_models.py --sherpa paraformer-en
```

## 3. ✅ Alignment Robustness (COMPLETED)

**Status**: Fully implemented

**Implementation**: [eval/align/align_words_to_rttm.py](eval/align/align_words_to_rttm.py)

### Features:

#### Epsilon Tolerance (fuzzy interval matching)
- Expands word boundaries by ±epsilon (default: 0.05s)
- Handles slight misalignment between ASR and diarization timestamps
- Configurable per run

#### Multi-speaker Overlap Handling
- Computes overlap for all speakers
- Selects speaker with maximum overlap
- **Tie Resolution**: Returns "UNK" if multiple speakers have equal overlap
- Avoids arbitrary assignment in ambiguous cases

#### Strict SAD Mode
- `strict_sad=true`: Drops words that fall outside all speech segments
- `strict_sad=false` (default): Labels such words as "UNKNOWN"
- Useful for filtering ASR hallucinations

#### Probability Preservation
- Preserves ASR confidence scores if available
- Falls back to 1.0 if not provided

### Configuration (eval.yaml):
```yaml
alignment:
  epsilon: 0.05        # tolerance for boundaries (seconds)
  strict_sad: false    # drop words outside speech segments
```

### CLI:
```bash
python eval/align/align_words_to_rttm.py \
  --asr outputs/asr/faster_whisper/clip.json \
  --rttm outputs/dia/pyannote/clip.rttm \
  --out aligned.jsonl \
  --epsilon 0.1 \
  --strict-sad
```

### Output Format:
```json
{"file":"clip","tool_asr":"fw","tool_dia":"pyannote","w":"Hello","start":12.34,"end":12.60,"speaker":"SPEAKER_01","prob":0.92}
{"file":"clip","tool_asr":"fw","tool_dia":"pyannote","w":"there","start":12.61,"end":12.95,"speaker":"UNK","prob":0.88}
```

## 4. ✅ Pinned Versions (COMPLETED)

**Status**: Fully pinned

**File**: [eval/requirements.txt](eval/requirements.txt)

### All versions now pinned for reproducibility:

```txt
pyyaml==6.0.1
numpy==1.26.3
faster-whisper==1.0.3
sherpa-onnx==1.10.16
pyannote.audio==3.1.1
speechbrain==1.0.0
torch==2.1.2
torchaudio==2.1.2
scikit-learn==1.4.0
dscore==0.1.2
huggingface-hub==0.20.3
```

### Installation:
```bash
pip install -r eval/requirements.txt
```

## 5. ✅ Environment Manifest Logging (COMPLETED)

**Status**: Fully implemented

**Implementation**: [eval/utils/env.py](eval/utils/env.py)

### Features:
- Captures full environment snapshot on each run
- Logs to `outputs/logs/env.json`
- Includes:
  - Python version
  - Platform info (OS, architecture, etc.)
  - All installed packages with versions
  - Git info (commit, branch, dirty status)

### Usage:
```python
from eval.utils import save_env_manifest, print_env_summary

# Save manifest
save_env_manifest("outputs/logs/env.json")

# Print summary
print_env_summary()
```

### Output Example (env.json):
```json
{
  "python_version": "3.11.7",
  "platform": {
    "system": "Darwin",
    "release": "23.2.0",
    "machine": "arm64"
  },
  "packages": {
    "torch": "2.1.2",
    "faster-whisper": "1.0.3",
    ...
  },
  "git": {
    "commit": "abc123...",
    "branch": "main",
    "dirty": false
  }
}
```

## 6. ⚠️ Runner Enhancements (PARTIAL)

**Status**: Basic alignment config support added

### Added to runner.py:
- Alignment config is now loaded from eval.yaml
- Environment manifest is saved at run start (future enhancement)

### Planned enhancements (not yet implemented):
```bash
# Specify tool pairs to run
python runner.py --pairs "asr=fw,sherpa dia=pyannote,speechbrain"

# Skip stages
python runner.py --skip 1,2  # Skip stages 1 and 2
```

**Implementation note**: These would require additional CLI parsing in runner.py. The current runner processes all enabled tools from config.

## 7. ⚠️ Scoring Enhancements (PARTIAL)

### WER Improvements:
- ✅ Levenshtein distance implementation (already present)
- ✅ Text normalization (case, punctuation)
- ✅ cpWER grouping by speaker
- ⚠️ STM format support - not yet implemented

### DER Improvements:
- ✅ dscore wrapper with error handling
- ✅ Fallback to simple Python DER if dscore unavailable
- ⚠️ File ID parity checking - basic implementation present

**STM Format**: To add STM support, extend `eval/score/wer.py`:

```python
def load_stm(stm_path: Path) -> Dict[str, str]:
    """Load STM reference file.

    STM format:
    <file_id> <channel> <speaker> <start> <end> <text>
    """
    # Implementation left for users who need STM support
    pass
```

## 8. ❌ CI Smoke Test (NOT IMPLEMENTED)

**Status**: Not implemented

**Reason**: Requires:
- Sample 30s WAV file
- GitHub Actions workflow
- Pre-downloaded models or mock inference

**Recommendation**: Add in production deployment. Template:

```yaml
# .github/workflows/test.yml
name: Smoke Test
on: [push, pull_request]

jobs:
  test:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r eval/requirements.txt
      - run: python eval/test_install.py
      # Add: python runner.py --stage 1 (with sample WAV)
```

## Acceptance Criteria Status

| Criterion | Status | Notes |
|-----------|--------|-------|
| SpeechBrain produces RTTM | ✅ | Full implementation with clustering |
| Sherpa model downloader | ✅ | Single command downloads & validates |
| Alignment epsilon & ties | ✅ | Epsilon tolerance + "UNK" for ties |
| Pinned versions | ✅ | All deps pinned in requirements.txt |
| Env manifest | ✅ | Saved to outputs/logs/env.json |
| --pairs flag | ⚠️ | Not implemented (use config instead) |
| CI test | ❌ | Not implemented |
| Documentation | ✅ | This file + updated README |

## Migration Guide

### For Existing Users:

1. **Update requirements**:
   ```bash
   pip install -r eval/requirements.txt --upgrade
   ```

2. **Update eval.yaml** to add new sections:
   ```yaml
   diarization:
     speechbrain:
       enabled: true  # was false by default
       cluster_method: "spectral"
       window_size: 1.5
       hop_size: 0.75

   alignment:
     epsilon: 0.05
     strict_sad: false
   ```

3. **Download sherpa models** (if using sherpa-onnx):
   ```bash
   python eval/utils/download_models.py --sherpa paraformer-en
   ```

4. **Update model_path** in eval.yaml:
   ```yaml
   asr:
     sherpa_onnx:
       enabled: true
       model_path: "models/sherpa-onnx/sherpa-onnx-paraformer-en-2023-10-24"
   ```

## Testing

### Test SpeechBrain:
```bash
python -c "
from eval.diar import SpeechBrainDiarization
d = SpeechBrainDiarization()
segments = d.diarize('data/clips/test.wav')
print(f'Found {len(segments)} segments')
"
```

### Test Model Downloader:
```bash
python eval/utils/download_models.py --list
python eval/utils/download_models.py --check paraformer-en
```

### Test Alignment:
```bash
python eval/align/align_words_to_rttm.py \
  --asr outputs/asr/faster_whisper/clip.json \
  --rttm outputs/dia/pyannote/clip.rttm \
  --out test_aligned.jsonl \
  --epsilon 0.1 \
  --strict-sad
```

### Test Env Logging:
```python
from eval.utils import print_env_summary
print_env_summary()
```

## Known Limitations

1. **SpeechBrain clustering**: Auto speaker-count detection may fail on very short clips (<30s). Use `oracle_n_speakers=true` for short clips.

2. **Sherpa-ONNX**: Models are large (100-500 MB). Ensure adequate disk space.

3. **Alignment epsilon**: Very large epsilon (>0.5s) may cause false positive alignments. Keep ≤0.1s for best results.

4. **CI testing**: Not automated. Manual testing required before deployment.

## Future Work

### Priority 1 (High Value):
- [ ] Add `--pairs` and `--skip` flags to runner.py
- [ ] Integrate env manifest saving into runner
- [ ] Add CI smoke test with sample WAV

### Priority 2 (Medium Value):
- [ ] STM format support for references
- [ ] Improved cpWER with speaker mapping
- [ ] WER/DER variance calculation (stability metric)

### Priority 3 (Nice to Have):
- [ ] GPU support toggle for faster-whisper
- [ ] Parallel clip processing
- [ ] Web UI for results visualization

## Questions?

See:
- [README.md](README.md) - Full documentation
- [QUICKSTART.md](QUICKSTART.md) - Quick setup guide
- [eval.yaml](eval.yaml) - Configuration reference

Or run:
```bash
python eval/test_install.py  # Verify installation
python eval/runner.py --help  # See all options
```
