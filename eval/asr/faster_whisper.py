"""faster-whisper ASR wrapper."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FasterWhisperASR:
    """Wrapper for faster-whisper ASR."""

    def __init__(
        self,
        model: str = "large-v3",
        compute_type: str = "int8_float16",
        device: str = "auto",
        vad_filter: bool = True,
        language: Optional[str] = "en",
        beam_size: int = 5,
    ):
        """Initialize faster-whisper ASR.

        Args:
            model: Model name (e.g., "large-v3", "medium.en")
            compute_type: Computation type for quantization
            device: Device to use ("cpu", "cuda", or "auto")
            vad_filter: Whether to use VAD filtering
            language: Language code (None for auto-detection)
            beam_size: Beam size for decoding
        """
        self.model_name = model
        self.compute_type = compute_type
        self.device = device
        self.vad_filter = vad_filter
        self.language = language
        self.beam_size = beam_size
        self.model = None

        logger.info(f"Initializing faster-whisper with model={model}, compute_type={compute_type}")

    def load_model(self) -> None:
        """Load the Whisper model."""
        if self.model is not None:
            return

        try:
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info(f"Loaded faster-whisper model: {self.model_name}")

        except ImportError:
            raise ImportError(
                "faster-whisper is not installed. Install with: pip install faster-whisper"
            )
        except Exception as e:
            logger.error(f"Failed to load faster-whisper model: {e}")
            raise

    def transcribe(self, audio_path: str | Path) -> Dict[str, Any]:
        """Transcribe audio file.

        Args:
            audio_path: Path to audio file (WAV)

        Returns:
            Dictionary with transcription results:
            {
                "file": str,
                "language": str,
                "segments": [
                    {
                        "start": float,
                        "end": float,
                        "text": str,
                        "words": [
                            {"w": str, "start": float, "end": float, "p": float}
                        ]
                    }
                ]
            }
        """
        if self.model is None:
            self.load_model()

        audio_path = Path(audio_path)
        logger.info(f"Transcribing {audio_path.name} with faster-whisper")

        try:
            segments, info = self.model.transcribe(
                str(audio_path),
                language=self.language,
                vad_filter=self.vad_filter,
                beam_size=self.beam_size,
                word_timestamps=True,
            )

            # Convert generator to list and extract data
            result_segments = []
            for seg in segments:
                segment_dict = {
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": seg.text.strip(),
                    "words": [],
                }

                # Extract word-level timestamps if available
                if seg.words:
                    for word in seg.words:
                        word_dict = {
                            "w": word.word.strip(),
                            "start": round(word.start, 3),
                            "end": round(word.end, 3),
                            "p": round(word.probability, 4),
                        }
                        segment_dict["words"].append(word_dict)

                result_segments.append(segment_dict)

            result = {
                "file": audio_path.name,
                "language": info.language if info.language else self.language or "en",
                "segments": result_segments,
            }

            logger.info(
                f"Transcription complete: {len(result_segments)} segments, "
                f"language={result['language']}"
            )

            return result

        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            raise

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


def run_faster_whisper(
    audio_path: str | Path, output_path: str | Path, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run faster-whisper ASR on an audio file.

    Args:
        audio_path: Path to input audio file
        output_path: Path to save JSON output
        config: Configuration dictionary

    Returns:
        Transcription result
    """
    asr_config = config.get("asr", {}).get("faster_whisper", {})

    asr = FasterWhisperASR(
        model=asr_config.get("model", "large-v3"),
        compute_type=asr_config.get("compute_type", "int8_float16"),
        vad_filter=asr_config.get("vad_filter", True),
        language=asr_config.get("language", "en"),
        beam_size=asr_config.get("beam_size", 5),
    )

    result = asr.transcribe(audio_path)

    # Save result
    from eval.utils import save_json

    save_json(result, output_path)

    return result
