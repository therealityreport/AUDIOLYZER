# Installing Audio Preprocessing

## Quick Install

```bash
# Method 1: Using the install script (recommended)
python scripts/install_preprocessing.py

# Method 2: Using pip
pip install -e ".[preprocessing]"

# Method 3: Manual install
pip install audio-separator resemble-enhance
```

## Verification

```bash
python -c "from audio_separator.separator import Separator; print('✓ audio-separator works')"
python -c "from resemble_enhance.enhancer.inference import enhance; print('✓ resemble-enhance works')"
```

> The install script performs these checks automatically; run them manually when using the pip/manual flows.

## Usage

### Enable in Config

```yaml
# configs/my_show.yaml
audio_preprocessing:
  enable: true
  retain_intermediates: true
  vocal_separation:
    enable: true
  enhancement:
    enable: true
```

> Base `configs/dev.yaml` and `configs/prod.yaml` keep `audio_preprocessing.enable: false`; merge this block via a show preset (e.g., `configs/reality_tv.yaml`) or pass `--preprocess` when you want to opt in.

### Process Episode

```bash
show-scribe process episode.mp4 --config configs/reality_tv.yaml

# Override once even if the config disables it
show-scribe process episode.mp4 --preprocess
```

## Storage Requirements

Preprocessing models will be downloaded to:
- `~/.cache/audio-separator/` (~2-3 GB)
- `~/.cache/resemble-enhance/` (~100 MB)

Total: ~3 GB additional storage needed

## Troubleshooting

### ImportError: No module named 'audio_separator'
```bash
pip install audio-separator
```

### ImportError: No module named 'resemble_enhance'
```bash
pip install resemble-enhance
```

### CUDA/GPU Issues
Preprocessing will work on CPU, but GPU is faster:
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Memory Errors
If you run out of memory:
1. Close other applications
2. Use smaller model: `mdx_extra` instead of `htdemucs`
3. Process shorter episodes (<30 min)

## Uninstall

```bash
pip uninstall audio-separator resemble-enhance
rm -rf ~/.cache/audio-separator ~/.cache/resemble-enhance
```
