"""DER (Diarization Error Rate) and JER (Jaccard Error Rate) scoring."""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def compute_der_with_dscore(
    reference_rttm: str | Path, hypothesis_rttm: str | Path, collar: float = 0.25
) -> Dict[str, Any]:
    """Compute DER using dscore tool.

    Args:
        reference_rttm: Path to reference RTTM
        hypothesis_rttm: Path to hypothesis RTTM
        collar: Collar size in seconds (tolerance for boundary errors)

    Returns:
        Dictionary with DER metrics
    """
    try:
        # Run dscore command
        cmd = [
            "dscore",
            "-r",
            str(reference_rttm),
            "-s",
            str(hypothesis_rttm),
            "--collar",
            str(collar),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse output
        output = result.stdout
        metrics = parse_dscore_output(output)

        logger.info(f"DER computed: {metrics.get('DER', 'N/A')}")

        return metrics

    except FileNotFoundError:
        logger.error(
            "dscore not found. Install with: pip install dscore\n"
            "or use the pure Python implementation"
        )
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"dscore failed: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Failed to compute DER: {e}")
        raise


def parse_dscore_output(output: str) -> Dict[str, Any]:
    """Parse dscore output.

    Args:
        output: dscore stdout text

    Returns:
        Dictionary with parsed metrics
    """
    metrics = {}

    for line in output.split("\n"):
        line = line.strip()

        # Look for key metrics
        if "DER" in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "DER" and i + 1 < len(parts):
                    try:
                        metrics["DER"] = float(parts[i + 1].rstrip("%"))
                    except ValueError:
                        pass

        if "Miss" in line or "FA" in line or "Confusion" in line:
            # Parse detailed metrics if available
            pass

    return metrics


def compute_der_simple(
    reference_segments: List[Tuple[float, float, str]],
    hypothesis_segments: List[Tuple[float, float, str]],
    total_duration: float,
    collar: float = 0.25,
) -> Dict[str, Any]:
    """Compute DER using simple Python implementation.

    DER = (False Alarm + Missed Detection + Speaker Confusion) / Total Speech Time

    Args:
        reference_segments: List of (start, end, speaker) tuples
        hypothesis_segments: List of (start, end, speaker) tuples
        total_duration: Total audio duration in seconds
        collar: Collar size in seconds

    Returns:
        Dictionary with DER metrics
    """
    # This is a simplified implementation
    # For production, use dscore or pyannote.metrics

    # Convert segments to frame-level labels
    frame_rate = 100  # 10ms frames
    num_frames = int(total_duration * frame_rate)

    ref_frames = [""] * num_frames
    hyp_frames = [""] * num_frames

    # Fill reference frames
    for start, end, speaker in reference_segments:
        start_frame = int(start * frame_rate)
        end_frame = int(end * frame_rate)
        for i in range(start_frame, min(end_frame, num_frames)):
            ref_frames[i] = speaker

    # Fill hypothesis frames
    for start, end, speaker in hypothesis_segments:
        start_frame = int(start * frame_rate)
        end_frame = int(end * frame_rate)
        for i in range(start_frame, min(end_frame, num_frames)):
            hyp_frames[i] = speaker

    # Count errors
    false_alarm = 0  # Hyp speech where Ref is non-speech
    missed_speech = 0  # Ref speech where Hyp is non-speech
    speaker_error = 0  # Both speech but wrong speaker

    total_ref_speech = 0

    for i in range(num_frames):
        ref_spk = ref_frames[i]
        hyp_spk = hyp_frames[i]

        if ref_spk:  # Reference has speech
            total_ref_speech += 1

            if not hyp_spk:  # Hypothesis missed it
                missed_speech += 1
            elif ref_spk != hyp_spk:  # Wrong speaker
                speaker_error += 1

        elif hyp_spk:  # Reference has no speech but hypothesis does
            false_alarm += 1

    # Calculate DER
    if total_ref_speech == 0:
        der = 0.0
    else:
        der = (false_alarm + missed_speech + speaker_error) / total_ref_speech

    return {
        "DER": round(der * 100, 2),  # as percentage
        "DER_raw": round(der, 4),
        "false_alarm": false_alarm,
        "missed_speech": missed_speech,
        "speaker_error": speaker_error,
        "total_frames": num_frames,
        "total_ref_speech_frames": total_ref_speech,
        "note": "Simple frame-based DER calculation",
    }


def compute_jer(
    reference_segments: List[Tuple[float, float, str]],
    hypothesis_segments: List[Tuple[float, float, str]],
) -> float:
    """Compute Jaccard Error Rate (JER).

    JER = 1 - (Intersection / Union) of speaker segments

    Args:
        reference_segments: List of (start, end, speaker) tuples
        hypothesis_segments: List of (start, end, speaker) tuples

    Returns:
        JER value (0 = perfect, 1 = completely wrong)
    """
    # Simplified JER calculation
    # A full implementation would consider speaker label mapping

    # For now, compute overlap-based metric
    ref_set = set()
    hyp_set = set()

    # Create time-speaker tuples (discretized)
    for start, end, speaker in reference_segments:
        for t in range(int(start * 10), int(end * 10)):  # 100ms resolution
            ref_set.add((t, speaker))

    for start, end, speaker in hypothesis_segments:
        for t in range(int(start * 10), int(end * 10)):
            hyp_set.add((t, speaker))

    # Jaccard similarity
    if not ref_set and not hyp_set:
        return 0.0

    intersection = len(ref_set & hyp_set)
    union = len(ref_set | hyp_set)

    jaccard = intersection / union if union > 0 else 0.0
    jer = 1.0 - jaccard

    return round(jer, 4)


def compute_diarization_metrics(
    reference_rttm: str | Path,
    hypothesis_rttm: str | Path,
    total_duration: Optional[float] = None,
    use_dscore: bool = True,
) -> Dict[str, Any]:
    """Compute diarization metrics (DER and JER).

    Args:
        reference_rttm: Path to reference RTTM
        hypothesis_rttm: Path to hypothesis RTTM
        total_duration: Total audio duration (required for simple DER)
        use_dscore: Whether to use dscore tool (if available)

    Returns:
        Dictionary with diarization metrics
    """
    from eval.utils import load_rttm

    metrics = {}

    # Try dscore first if requested
    if use_dscore:
        try:
            metrics = compute_der_with_dscore(reference_rttm, hypothesis_rttm)
            metrics["method"] = "dscore"
        except Exception as e:
            logger.warning(f"dscore failed, falling back to simple DER: {e}")
            use_dscore = False

    # Fall back to simple implementation
    if not use_dscore or not metrics:
        if total_duration is None:
            logger.error("total_duration required for simple DER calculation")
            return {"error": "total_duration required"}

        ref_segments = load_rttm(reference_rttm)
        hyp_segments = load_rttm(hypothesis_rttm)

        metrics = compute_der_simple(ref_segments, hyp_segments, total_duration)
        metrics["method"] = "simple"

        # Also compute JER
        jer = compute_jer(ref_segments, hyp_segments)
        metrics["JER"] = jer

    return metrics


def batch_compute_der(
    reference_rttms: Dict[str, str | Path],
    hypothesis_rttms: Dict[str, str | Path],
    durations: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute DER for multiple clips.

    Args:
        reference_rttms: Dictionary mapping clip names to reference RTTM paths
        hypothesis_rttms: Dictionary mapping clip names to hypothesis RTTM paths
        durations: Optional dictionary of clip durations

    Returns:
        Dictionary mapping clip names to DER metrics
    """
    results = {}

    for clip_name in reference_rttms:
        if clip_name not in hypothesis_rttms:
            logger.warning(f"No hypothesis RTTM found for {clip_name}, skipping")
            continue

        ref_path = reference_rttms[clip_name]
        hyp_path = hypothesis_rttms[clip_name]
        duration = durations.get(clip_name) if durations else None

        try:
            metrics = compute_diarization_metrics(ref_path, hyp_path, total_duration=duration)
            results[clip_name] = metrics
            logger.info(f"{clip_name}: DER = {metrics.get('DER', 'N/A')}%")

        except Exception as e:
            logger.error(f"Failed to compute DER for {clip_name}: {e}")
            results[clip_name] = {"error": str(e)}

    return results
