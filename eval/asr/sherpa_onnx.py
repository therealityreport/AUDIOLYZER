"""sherpa-onnx ASR wrapper."""

import logging
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SherpaOnnxASR:
    """Wrapper for sherpa-onnx ASR."""

    def __init__(
        self,
        model_path: str | Path,
        model_type: str = "paraformer",
        tokens: Optional[str] = None,
        num_threads: int = 4,
    ):
        """Initialize sherpa-onnx ASR.

        Args:
            model_path: Path to model directory
            model_type: Model type ("paraformer" or "zipformer")
            tokens: Path to tokens file (relative to model_path if not absolute)
            num_threads: Number of threads for inference
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.num_threads = num_threads
        self.recognizer = None

        # Resolve tokens path
        if tokens:
            tokens_path = Path(tokens)
            if not tokens_path.is_absolute():
                tokens_path = self.model_path / tokens
            self.tokens = str(tokens_path)
        else:
            self.tokens = str(self.model_path / "tokens.txt")

        logger.info(f"Initializing sherpa-onnx with model_path={model_path}, type={model_type}")

    def _validate_model_files(self) -> None:
        """Validate that required model files exist."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_path}\n"
                f"Download with: python eval/utils/download_models.py --sherpa {self.model_type}-en"
            )

        # Check tokens file
        if not Path(self.tokens).exists():
            raise FileNotFoundError(
                f"tokens.txt not found: {self.tokens}\n"
                f"Expected in: {self.model_path}\n"
                f"Download model with: python eval/utils/download_models.py --sherpa {self.model_type}-en"
            )

        # Check model-specific files
        if self.model_type == "paraformer":
            model_file = self.model_path / "model.onnx"
            if not model_file.exists():
                model_file = self.model_path / "paraformer.onnx"
            if not model_file.exists():
                raise FileNotFoundError(
                    f"Paraformer model.onnx not found in: {self.model_path}\n"
                    f"Download with: python eval/utils/download_models.py --sherpa paraformer-en"
                )

        elif self.model_type == "zipformer":
            required = ["encoder.onnx", "decoder.onnx", "joiner.onnx"]
            missing = [f for f in required if not (self.model_path / f).exists()]
            if missing:
                raise FileNotFoundError(
                    f"Zipformer model files missing: {', '.join(missing)}\n"
                    f"Expected in: {self.model_path}\n"
                    f"Download with: python eval/utils/download_models.py --sherpa zipformer-en"
                )

    def load_model(self) -> None:
        """Load the sherpa-onnx model."""
        if self.recognizer is not None:
            return

        # Validate files first
        self._validate_model_files()

        try:
            import sherpa_onnx

            if self.model_type == "paraformer":
                # Paraformer configuration
                model_file = self.model_path / "model.onnx"
                if not model_file.exists():
                    model_file = self.model_path / "paraformer.onnx"

                config = sherpa_onnx.OfflineRecognizerConfig(
                    model_config=sherpa_onnx.OfflineModelConfig(
                        paraformer=sherpa_onnx.OfflineParaformerModelConfig(model=str(model_file)),
                        tokens=self.tokens,
                        num_threads=self.num_threads,
                    )
                )

            elif self.model_type == "zipformer":
                # Zipformer/transducer configuration
                encoder = self.model_path / "encoder.onnx"
                decoder = self.model_path / "decoder.onnx"
                joiner = self.model_path / "joiner.onnx"

                config = sherpa_onnx.OfflineRecognizerConfig(
                    model_config=sherpa_onnx.OfflineModelConfig(
                        transducer=sherpa_onnx.OfflineTransducerModelConfig(
                            encoder=str(encoder),
                            decoder=str(decoder),
                            joiner=str(joiner),
                        ),
                        tokens=self.tokens,
                        num_threads=self.num_threads,
                    )
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            self.recognizer = sherpa_onnx.OfflineRecognizer(config)
            logger.info(f"Loaded sherpa-onnx model: {self.model_type}")

        except ImportError:
            raise ImportError("sherpa-onnx is not installed. Install with: pip install sherpa-onnx")
        except Exception as e:
            logger.error(f"Failed to load sherpa-onnx model: {e}")
            raise

    def _read_wave(self, audio_path: str | Path) -> tuple:
        """Read WAV file and return samples and sample rate.

        Args:
            audio_path: Path to WAV file

        Returns:
            (samples, sample_rate) tuple
        """
        with wave.open(str(audio_path), "rb") as wf:
            assert wf.getnchannels() == 1, "Only mono audio is supported"
            assert wf.getsampwidth() == 2, "Only 16-bit audio is supported"

            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        return samples, sample_rate

    def transcribe(self, audio_path: str | Path) -> Dict[str, Any]:
        """Transcribe audio file.

        Args:
            audio_path: Path to audio file (WAV)

        Returns:
            Dictionary with transcription results
        """
        if self.recognizer is None:
            self.load_model()

        audio_path = Path(audio_path)
        logger.info(f"Transcribing {audio_path.name} with sherpa-onnx")

        try:
            # Read audio
            samples, sample_rate = self._read_wave(audio_path)

            # Create stream
            stream = self.recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)

            # Decode
            self.recognizer.decode_stream(stream)
            result_text = stream.result.text

            # sherpa-onnx doesn't provide word-level timestamps by default
            # We create a single segment with the full text
            # Duration estimation based on audio length
            duration = len(samples) / sample_rate

            segments = []
            if result_text.strip():
                # Split into rough segments by sentence boundaries
                sentences = self._split_sentences(result_text)
                segment_duration = duration / len(sentences) if sentences else duration

                for i, sentence in enumerate(sentences):
                    start = i * segment_duration
                    end = (i + 1) * segment_duration

                    segment = {
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "text": sentence.strip(),
                        "words": self._estimate_word_timestamps(sentence, start, end),
                    }
                    segments.append(segment)

            result = {
                "file": audio_path.name,
                "language": "en",  # sherpa-onnx models are typically language-specific
                "segments": segments,
                "note": "Word timestamps are estimated (sherpa-onnx doesn't provide them)",
            }

            logger.info(f"Transcription complete: {len(segments)} segments")

            return result

        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            raise

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        import re

        # Simple sentence splitter
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _estimate_word_timestamps(
        self, text: str, start: float, end: float
    ) -> List[Dict[str, Any]]:
        """Estimate word-level timestamps.

        This is a rough estimation since sherpa-onnx doesn't provide
        word-level timestamps by default.

        Args:
            text: Segment text
            start: Segment start time
            end: Segment end time

        Returns:
            List of word dictionaries
        """
        words = text.split()
        if not words:
            return []

        duration = end - start
        word_duration = duration / len(words)

        word_list = []
        for i, word in enumerate(words):
            word_start = start + i * word_duration
            word_end = word_start + word_duration

            word_dict = {
                "w": word,
                "start": round(word_start, 3),
                "end": round(word_end, 3),
                "p": 1.0,  # No confidence score available
            }
            word_list.append(word_dict)

        return word_list

    def transcribe_batch(self, audio_paths: List[str | Path]) -> Dict[str, Dict[str, Any]]:
        """Transcribe multiple audio files.

        Args:
            audio_paths: List of audio file paths

        Returns:
            Dictionary mapping file names to transcription results
        """
        results = {}

        for audio_path in audio_paths:
            audio_path = Path(audio_path)
            try:
                result = self.transcribe(audio_path)
                results[audio_path.stem] = result
            except Exception as e:
                logger.error(f"Failed to transcribe {audio_path.name}: {e}")
                results[audio_path.stem] = {"error": str(e)}

        return results


def run_sherpa_onnx(
    audio_path: str | Path, output_path: str | Path, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run sherpa-onnx ASR on an audio file.

    Args:
        audio_path: Path to input audio file
        output_path: Path to save JSON output
        config: Configuration dictionary

    Returns:
        Transcription result
    """
    asr_config = config.get("asr", {}).get("sherpa_onnx", {})

    model_path = asr_config.get("model_path", "models/sherpa-onnx/paraformer")
    model_type = asr_config.get("model_type", "paraformer")
    tokens = asr_config.get("tokens")

    asr = SherpaOnnxASR(
        model_path=model_path,
        model_type=model_type,
        tokens=tokens,
    )

    result = asr.transcribe(audio_path)

    # Save result
    from eval.utils import save_json

    save_json(result, output_path)

    return result
