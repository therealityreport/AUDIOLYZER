# Audio Enhancement Stack

| Provider      | Components                              | Best For                         |
|---------------|------------------------------------------|----------------------------------|
| `resemble`    | Demucs (`audio-separator`) + `resemble-enhance` | Default runs, fast turnarounds, offline GPUs |
| `clearervoice`| ClearerVoice Studio (`clearvoice` package) | Quality mode, heavy noise, fallback when Demucs struggles |

- Toggle via `audio_preprocessing.enhancement.provider` or the Audio Preprocessing dropdown in the Streamlit UI.
- ClearerVoice yields both `audio_enhanced_vocals.wav` and `audio_enhanced_mix.wav`; retain `audio_enhanced_mix.wav` for review teams that prefer ambience preserved.
- Install dependencies with `python scripts/install_preprocessing.py` or `pip install -e '.[preprocessing]'`.
- Capture runtime/WER deltas with `scripts/bench_clearervoice_vs_current.py` to monitor regressions as models update.
