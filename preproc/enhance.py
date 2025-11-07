#!/usr/bin/env python3
"""Audio enhancement module for preprocessing."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict
import json
import os

try:
    from resemble_enhance.enhancer.inference import enhance
except ImportError as e:
    print(
        f"Error: resemble-enhance not installed. Run: pip install resemble-enhance", file=sys.stderr
    )
    sys.exit(1)


def load_config(preset_path: Path) -> Dict[str, Any]:
    """Load configuration from preset YAML file."""
    import yaml

    with open(preset_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config.get("audio_preprocessing", {}).get("enhancement", {})


def main():
    """Main entry point for audio enhancement."""
    parser = argparse.ArgumentParser(
        description="Enhance audio quality with denoising and dereverberation"
    )
    parser.add_argument("--preset", required=True, help="Path to preset YAML file")
    parser.add_argument("--input", required=True, help="Input audio file path")
    parser.add_argument("--output", required=True, help="Output enhanced file path")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    preset_path = Path(args.preset)

    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not preset_path.exists():
        print(f"Error: Preset file does not exist: {preset_path}", file=sys.stderr)
        sys.exit(1)

    # Load configuration
    config = load_config(preset_path)
    denoise = config.get("denoise", True)
    dereverb = config.get("dereverb", True)
    nfe = int(config.get("nfe", 64) or 64)
    lambd = float(config.get("lambd", 0.7))
    tau = float(config.get("tau", 0.6))
    chunk_seconds = float(config.get("chunk_seconds", 18.0) or 18.0)
    target_sample_rate = int(config.get("target_sample_rate", 16000) or 16000)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Enhancing audio with denoise={denoise}, dereverb={dereverb}")
        print(f"Parameters: nfe={nfe}, lambd={lambd}, tau={tau}")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")

        # Load audio file
        import soundfile as sf
        import torch
        import numpy as np

        audio_data, sample_rate = sf.read(str(input_path))

        # Convert numpy array to torch tensor (required by resemble-enhance)
        # Ensure audio is in the correct shape (samples,) for mono or (samples, channels) for stereo
        if audio_data.ndim == 1:
            # Mono audio
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
        else:
            # Stereo or multi-channel - convert to mono by averaging channels
            audio_mono = np.mean(audio_data, axis=1).astype(np.float32)
            audio_tensor = torch.from_numpy(audio_mono)

        print(f"Audio shape: {audio_tensor.shape}, sample rate: {sample_rate}")

        # Enhance audio
        import torchaudio

        # Pick a device automatically (CUDA -> CPU; MPS requires explicit opt-in).
        allow_mps = os.getenv("SHOW_SCRIBE_ALLOW_MPS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if torch.cuda.is_available():
            device = "cuda"
        elif (
            allow_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ):
            device = "mps"
        else:
            device = "cpu"
            if (
                not torch.cuda.is_available()
                and getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
                and not allow_mps
            ):
                print(
                    "MPS detected but disabled to avoid unsupported ops; using CPU fallback.",
                    file=sys.stderr,
                )

        # On CPU/MPS the diffusion solver is extremely slow; clamp to a lighter setting.
        if device != "cuda":
            nfe = max(8, min(nfe, 24))

        # Resample to target sample rate to keep inference affordable.
        if sample_rate != target_sample_rate:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor.unsqueeze(0), sample_rate, target_sample_rate
            ).squeeze(0)
            sample_rate = target_sample_rate
            print(f"Resampled audio to {sample_rate} Hz for enhancement.")

        # Chunk long clips to avoid runaway inference time on CPU/MPS.
        chunk_size = int(sample_rate * chunk_seconds)
        if chunk_size <= 0:
            chunk_size = int(sample_rate * 18.0)

        def _run_enhance(chunk: torch.Tensor) -> tuple[torch.Tensor, int]:
            enhanced_chunk, sr_out = enhance(
                chunk,
                sample_rate,
                device=device,
                nfe=nfe,
                solver="midpoint",
                lambd=lambd,
                tau=tau,
            )
            if not isinstance(enhanced_chunk, torch.Tensor):
                enhanced_chunk = torch.from_numpy(enhanced_chunk)
            if not isinstance(sr_out, int):
                sr_out = sample_rate
            return enhanced_chunk, sr_out

        if audio_tensor.numel() > chunk_size:
            print(
                f"Enhancing audio in {chunk_seconds:.1f}s chunks on {device} "
                f"(nfe={nfe}, lambd={lambd}, tau={tau})."
            )
            chunks: list[torch.Tensor] = []
            chunk_sample_rate = sample_rate
            for start in range(0, audio_tensor.numel(), chunk_size):
                segment = audio_tensor[start : start + chunk_size]
                enhanced_segment, sr_out = _run_enhance(segment)
                if sr_out != target_sample_rate:
                    enhanced_segment = torchaudio.functional.resample(
                        enhanced_segment.unsqueeze(0),
                        sr_out,
                        target_sample_rate,
                    ).squeeze(0)
                    sr_out = target_sample_rate
                chunk_sample_rate = sr_out
                chunks.append(enhanced_segment)
            enhanced_tensor = torch.cat(chunks, dim=0)
            sample_rate = chunk_sample_rate
        else:
            print(
                f"Enhancing audio in a single pass on {device} (nfe={nfe}, lambd={lambd}, tau={tau})."
            )
            enhanced_tensor, sr_out = _run_enhance(audio_tensor)
            if sr_out != target_sample_rate:
                enhanced_tensor = torchaudio.functional.resample(
                    enhanced_tensor.unsqueeze(0),
                    sr_out,
                    target_sample_rate,
                ).squeeze(0)
                sample_rate = target_sample_rate
            else:
                sample_rate = sr_out

        enhanced_audio = enhanced_tensor.cpu().numpy()
        enhanced_sr = sample_rate

        # Convert back to numpy for saving
        if isinstance(enhanced_audio, torch.Tensor):
            enhanced_audio = enhanced_audio.cpu().numpy()

        # Save enhanced audio
        sf.write(str(output_path), enhanced_audio, enhanced_sr)

        print(f"✓ Audio enhanced successfully: {output_path}")

    except Exception as e:
        import traceback

        print(f"Error during audio enhancement: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
