"""SpeechBrain diarization wrapper."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


class SpeechBrainDiarization:
    """Wrapper for SpeechBrain speaker diarization using ECAPA-TDNN embeddings."""

    def __init__(
        self,
        source: str = "speechbrain/spkrec-ecapa-voxceleb",
        oracle_n_speakers: bool = False,
        num_speakers: Optional[int] = None,
        cluster_method: str = "spectral",
        threshold: float = 0.5,
        window_size: float = 1.5,
        hop_size: float = 0.75,
    ):
        """Initialize SpeechBrain diarization.

        Args:
            source: Model source for ECAPA embeddings (Hugging Face or local)
            oracle_n_speakers: Whether to use oracle number of speakers
            num_speakers: Number of speakers (if oracle_n_speakers is True)
            cluster_method: Clustering method ("spectral" or "agglomerative")
            threshold: Similarity threshold for clustering
            window_size: Window size in seconds for embedding extraction
            hop_size: Hop size in seconds for sliding window
        """
        self.source = source
        self.oracle_n_speakers = oracle_n_speakers
        self.num_speakers = num_speakers
        self.cluster_method = cluster_method
        self.threshold = threshold
        self.window_size = window_size
        self.hop_size = hop_size
        self.embedding_model = None

        logger.info(f"Initializing SpeechBrain with source={source}, method={cluster_method}")

    def load_model(self) -> None:
        """Load the ECAPA embedding model."""
        if self.embedding_model is not None:
            return

        try:
            from speechbrain.pretrained import EncoderClassifier

            self.embedding_model = EncoderClassifier.from_hparams(
                source=self.source,
                savedir=f"tmpdir_sb/{self.source.replace('/', '_')}",
                run_opts={"device": "cpu"},  # Use CPU by default
            )

            logger.info(f"Loaded SpeechBrain ECAPA model: {self.source}")

        except ImportError:
            raise ImportError("SpeechBrain is not installed. Install with: pip install speechbrain")
        except Exception as e:
            logger.error(f"Failed to load SpeechBrain model: {e}")
            raise

    def _extract_embeddings(self, audio_path: str | Path) -> Tuple[np.ndarray, List[float]]:
        """Extract speaker embeddings using sliding window.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (embeddings array, timestamps list)
        """
        # Load audio
        waveform, sample_rate = torchaudio.load(str(audio_path))

        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Calculate window parameters
        window_samples = int(self.window_size * sample_rate)
        hop_samples = int(self.hop_size * sample_rate)

        # Extract embeddings for each window
        embeddings = []
        timestamps = []

        for start_sample in range(0, waveform.shape[1] - window_samples + 1, hop_samples):
            end_sample = start_sample + window_samples
            window = waveform[:, start_sample:end_sample]

            # Extract embedding
            with torch.no_grad():
                embedding = self.embedding_model.encode_batch(window)
                embeddings.append(embedding.squeeze().cpu().numpy())

            # Store timestamp (midpoint of window)
            timestamp = (start_sample + window_samples / 2) / sample_rate
            timestamps.append(timestamp)

        return np.array(embeddings), timestamps

    def _cluster_embeddings(
        self, embeddings: np.ndarray, n_speakers: Optional[int] = None
    ) -> np.ndarray:
        """Cluster embeddings to identify speakers.

        Args:
            embeddings: Array of speaker embeddings
            n_speakers: Number of speakers (if known)

        Returns:
            Array of cluster labels
        """
        from sklearn.cluster import AgglomerativeClustering, SpectralClustering
        from sklearn.metrics import silhouette_score

        if len(embeddings) == 0:
            return np.array([])

        # Compute similarity matrix (cosine similarity)
        from sklearn.metrics.pairwise import cosine_similarity

        similarity = cosine_similarity(embeddings)

        # Determine number of speakers
        if n_speakers is None:
            # Try different numbers of speakers and pick best silhouette score
            best_score = -1
            best_labels = None

            for k in range(2, min(len(embeddings), 10)):
                if self.cluster_method == "spectral":
                    clusterer = SpectralClustering(
                        n_clusters=k, affinity="precomputed", random_state=42
                    )
                else:  # agglomerative
                    clusterer = AgglomerativeClustering(
                        n_clusters=k, affinity="precomputed", linkage="average"
                    )

                labels = clusterer.fit_predict(similarity)

                # Compute silhouette score
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(similarity, labels, metric="precomputed")
                    if score > best_score:
                        best_score = score
                        best_labels = labels

            labels = (
                best_labels if best_labels is not None else np.zeros(len(embeddings), dtype=int)
            )

        else:
            # Use specified number of speakers
            if self.cluster_method == "spectral":
                clusterer = SpectralClustering(
                    n_clusters=n_speakers, affinity="precomputed", random_state=42
                )
            else:
                clusterer = AgglomerativeClustering(
                    n_clusters=n_speakers, affinity="precomputed", linkage="average"
                )

            labels = clusterer.fit_predict(similarity)

        return labels

    def _labels_to_segments(
        self, labels: np.ndarray, timestamps: List[float]
    ) -> List[Tuple[float, float, str]]:
        """Convert frame-level labels to speaker segments.

        Args:
            labels: Cluster labels for each frame
            timestamps: Timestamps for each frame

        Returns:
            List of (start, end, speaker_id) tuples
        """
        if len(labels) == 0:
            return []

        segments = []
        current_speaker = labels[0]
        start_time = max(0, timestamps[0] - self.hop_size / 2)

        for i in range(1, len(labels)):
            if labels[i] != current_speaker:
                # End current segment
                end_time = timestamps[i - 1] + self.hop_size / 2
                segments.append((start_time, end_time, f"SPEAKER_{current_speaker:02d}"))

                # Start new segment
                current_speaker = labels[i]
                start_time = timestamps[i] - self.hop_size / 2

        # Add final segment
        end_time = timestamps[-1] + self.hop_size / 2
        segments.append((start_time, end_time, f"SPEAKER_{current_speaker:02d}"))

        return segments

    def diarize(self, audio_path: str | Path) -> List[Tuple[float, float, str]]:
        """Perform speaker diarization.

        Args:
            audio_path: Path to audio file

        Returns:
            List of (start, end, speaker_id) tuples
        """
        if self.embedding_model is None:
            self.load_model()

        audio_path = Path(audio_path)
        logger.info(f"Diarizing {audio_path.name} with SpeechBrain ECAPA")

        try:
            # Extract embeddings
            logger.debug("Extracting speaker embeddings...")
            embeddings, timestamps = self._extract_embeddings(audio_path)

            if len(embeddings) == 0:
                logger.warning("No embeddings extracted")
                return []

            # Cluster embeddings
            logger.debug(f"Clustering {len(embeddings)} embeddings...")
            n_speakers = self.num_speakers if self.oracle_n_speakers else None
            labels = self._cluster_embeddings(embeddings, n_speakers)

            # Convert to segments
            segments = self._labels_to_segments(labels, timestamps)

            logger.info(
                f"Diarization complete: {len(segments)} segments, "
                f"{len(np.unique(labels))} speakers"
            )

            return segments

        except Exception as e:
            logger.error(f"Diarization failed for {audio_path}: {e}")
            raise

    def diarize_to_rttm(
        self, audio_path: str | Path, output_path: str | Path
    ) -> List[Tuple[float, float, str]]:
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
        self, audio_paths: List[str | Path], output_dir: Optional[str | Path] = None
    ) -> Dict[str, List[Tuple[float, float, str]]]:
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


def run_speechbrain(
    audio_path: str | Path, output_path: str | Path, config: Dict[str, Any]
) -> List[Tuple[float, float, str]]:
    """Run SpeechBrain diarization on an audio file.

    Args:
        audio_path: Path to input audio file
        output_path: Path to save RTTM output
        config: Configuration dictionary

    Returns:
        List of (start, end, speaker_id) tuples
    """
    diar_config = config.get("diarization", {}).get("speechbrain", {})

    diarizer = SpeechBrainDiarization(
        source=diar_config.get("source", "speechbrain/spkrec-ecapa-voxceleb"),
        oracle_n_speakers=diar_config.get("oracle_n_speakers", False),
        num_speakers=diar_config.get("num_speakers"),
        cluster_method=diar_config.get("cluster_method", "spectral"),
        threshold=diar_config.get("threshold", 0.5),
        window_size=diar_config.get("window_size", 1.5),
        hop_size=diar_config.get("hop_size", 0.75),
    )

    segments = diarizer.diarize_to_rttm(audio_path, output_path)

    return segments
