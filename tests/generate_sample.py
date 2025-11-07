#!/usr/bin/env python3
"""Generate a 30-second sample WAV file for testing."""

import math
import random
import struct
import wave


def generate_sample_audio(output_path: str, duration: int = 30, sample_rate: int = 16000):
    """Generate a simple test audio file.

    Creates a 30-second mono WAV with speech-like tones.

    Args:
        output_path: Output WAV file path
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
    """
    num_samples = duration * sample_rate
    samples = []

    # Generate speech-like signal
    for i in range(num_samples):
        t = i / sample_rate

        # Vary frequency over time to simulate different "words"
        segment_num = int(t / 3)  # 3-second segments
        f0 = 200 + (segment_num % 5) * 50  # 200-400 Hz range

        # Create speech-like signal with fundamental and harmonics
        signal = (
            0.3 * math.sin(2 * math.pi * f0 * t)
            + 0.2 * math.sin(2 * math.pi * 2 * f0 * t)
            + 0.1 * math.sin(2 * math.pi * 3 * f0 * t)
        )

        # Apply envelope within each segment
        seg_time = t % 3
        envelope = math.exp(-3 * seg_time)
        signal *= envelope

        # Add small noise
        noise = random.gauss(0, 0.02)
        signal += noise

        samples.append(signal)

    # Normalize
    max_val = max(abs(s) for s in samples)
    if max_val > 0:
        samples = [s / max_val for s in samples]

    # Convert to 16-bit PCM
    audio_data = b""
    for sample in samples:
        # Clamp to [-1, 1] and convert to 16-bit int
        sample = max(-1, min(1, sample))
        sample_int = int(sample * 32767)
        audio_data += struct.pack("<h", sample_int)

    # Write WAV file
    with wave.open(output_path, "w") as wf:
        wf.setnchannels(1)  # mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)

    print(f"Generated {duration}s sample audio: {output_path}")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Channels: 1 (mono)")
    print(f"  Bit depth: 16-bit PCM")
    print(f"  File size: {len(audio_data) / 1024:.1f} KB")


if __name__ == "__main__":
    import sys

    output_path = "tests/data/sample.wav"
    if len(sys.argv) > 1:
        output_path = sys.argv[1]

    generate_sample_audio(output_path, duration=30)
