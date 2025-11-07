"""pyannote.audio diarization wrapper."""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PyAnnoteDiarization:
    """Wrapper for pyannote.audio speaker diarization."""

    def __init__(
        self,
        pipeline: str = "pyannote/speaker-diarization@2.1",
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        segmentation_onset: float = 0.5,
        clustering_threshold: float = 0.7,
    ):
        """Initialize pyannote diarization.

        Args:
            pipeline: Hugging Face pipeline name
            min_speakers: Minimum number of speakers (None for auto)
            max_speakers: Maximum number of speakers (None for auto)
            segmentation_onset: Onset threshold for speech activity detection
            clustering_threshold: Threshold for speaker clustering
        """
        self.pipeline_name = pipeline
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.segmentation_onset = segmentation_onset
        self.clustering_threshold = clustering_threshold
        self.pipeline = None

        logger.info(f"Initializing pyannote with pipeline={pipeline}")

    def load_pipeline(self) -> None:
        """Load the diarization pipeline."""
        if self.pipeline is not None:
            return

        try:
            from pyannote.audio import Pipeline

            self.pipeline = Pipeline.from_pretrained(self.pipeline_name)

            logger.info(f"Loaded pyannote pipeline: {self.pipeline_name}")

        except ImportError:
            raise ImportError(
                "pyannote.audio is not installed. Install with: pip install pyannote.audio"
            ) from None
        except Exception as e:
            logger.error(f"Failed to load pyannote pipeline: {e}")
            logger.error(
                "You may need to authenticate with Hugging Face and accept model terms. "
                "Run: huggingface-cli login"
            )
            raise

    def diarize(self, audio_path: str | Path) -> list[tuple[float, float, str]]:
        """Perform speaker diarization.

        Args:
            audio_path: Path to audio file

        Returns:
            List of (start, end, speaker_id) tuples
        """
        if self.pipeline is None:
            self.load_pipeline()

        audio_path = Path(audio_path)
        logger.info(f"Diarizing {audio_path.name} with pyannote")

        try:
            # Prepare parameters
            params = {}

            if self.min_speakers is not None:
                params["min_speakers"] = self.min_speakers

            if self.max_speakers is not None:
                params["max_speakers"] = self.max_speakers

            # Run diarization
            diarization = self.pipeline(str(audio_path), **params)

            # Extract segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append((round(turn.start, 3), round(turn.end, 3), speaker))

            logger.info(f"Diarization complete: {len(segments)} segments")

            return segments

        except Exception as e:
            logger.error(f"Diarization failed for {audio_path}: {e}")
            raise

    def diarize_to_rttm(
        self, audio_path: str | Path, output_path: str | Path
    ) -> list[tuple[float, float, str]]:
        """Perform diarization and save as RTTM.

        Args:
            audio_path: Path to audio file
            output_path: Path to output RTTM file

        Returns:
            List of (start, end, speaker_id) tuples
        """
        segments = self.diarize(audio_path)

        # Save as RTTM
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        file_id = audio_path.stem

        with open(output_path, "w") as f:
            for start, end, speaker in segments:
                duration = end - start
                f.write(
                    f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} "
                    f"<NA> <NA> {speaker} <NA> <NA>\n"
                )

        logger.info(f"Saved RTTM to {output_path}")

        return segments

    def diarize_batch(
        self, audio_paths: list[str | Path], output_dir: str | Path | None = None
    ) -> dict[str, list[tuple[float, float, str]]]:
        """Diarize multiple audio files.

        Args:
            audio_paths: List of audio file paths
            output_dir: Optional directory to save RTTM files

        Returns:
            Dictionary mapping file names to segment lists
        """
        results = {}

        for audio_path in audio_paths:
            audio_path = Path(audio_path)

            try:
                if output_dir:
                    output_path = Path(output_dir) / f"{audio_path.stem}.rttm"
                    segments = self.diarize_to_rttm(audio_path, output_path)
                else:
                    segments = self.diarize(audio_path)

                results[audio_path.stem] = segments

            except Exception as e:
                logger.error(f"Failed to diarize {audio_path.name}: {e}")
                results[audio_path.stem] = []

        return results


def run_pyannote(
    audio_path: str | Path, output_path: str | Path, config: dict[str, Any]
) -> list[tuple[float, float, str]]:
    """Run pyannote diarization on an audio file.

    Args:
        audio_path: Path to input audio file
        output_path: Path to save RTTM output
        config: Configuration dictionary

    Returns:
        List of (start, end, speaker_id) tuples
    """
    diar_config = config.get("diarization", {}).get("pyannote", {})

    diarizer = PyAnnoteDiarization(
        pipeline=diar_config.get("pipeline", "pyannote/speaker-diarization@2.1"),
        min_speakers=diar_config.get("min_speakers"),
        max_speakers=diar_config.get("max_speakers"),
        segmentation_onset=diar_config.get("segmentation_onset", 0.5),
        clustering_threshold=diar_config.get("clustering_threshold", 0.7),
    )

    return diarizer.diarize_to_rttm(audio_path, output_path)
