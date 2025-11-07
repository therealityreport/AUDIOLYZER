"""Align ASR words to diarization speakers."""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def load_asr_words(asr_json_path: str | Path) -> List[Dict[str, Any]]:
    """Load words from ASR JSON.

    Args:
        asr_json_path: Path to ASR JSON file

    Returns:
        List of word dictionaries with start, end, word, probability
    """
    from eval.utils import load_json

    asr_data = load_json(asr_json_path)

    words = []
    for segment in asr_data.get("segments", []):
        for word in segment.get("words", []):
            words.append(
                {
                    "word": word.get("w", ""),
                    "start": word.get("start", 0.0),
                    "end": word.get("end", 0.0),
                    "probability": word.get("p", 1.0),
                }
            )

    return words


def load_rttm_segments(rttm_path: str | Path) -> List[Tuple[float, float, str]]:
    """Load speaker segments from RTTM.

    Args:
        rttm_path: Path to RTTM file

    Returns:
        List of (start, end, speaker_id) tuples
    """
    from eval.utils import load_rttm

    return load_rttm(rttm_path)


def find_speaker_for_word(
    word_start: float,
    word_end: float,
    speaker_segments: List[Tuple[float, float, str]],
    epsilon: float = 0.05,
    strict_sad: bool = False,
) -> Optional[str]:
    """Find which speaker is speaking during a word.

    Uses overlap-based assignment with epsilon tolerance: the speaker with
    the most overlap with the word's time span is assigned. In case of ties,
    returns "UNK".

    Args:
        word_start: Word start time
        word_end: Word end time
        speaker_segments: List of (start, end, speaker_id) tuples
        epsilon: Tolerance for interval boundaries (seconds)
        strict_sad: If True, return None for words outside all speech segments

    Returns:
        Speaker ID, "UNK" (for ties), or None (if no overlap and strict_sad)
    """
    # Expand word boundaries by epsilon for fuzzy matching
    word_start_expanded = word_start - epsilon
    word_end_expanded = word_end + epsilon

    overlaps = []

    for seg_start, seg_end, speaker_id in speaker_segments:
        # Calculate overlap with epsilon expansion
        overlap_start = max(word_start_expanded, seg_start)
        overlap_end = min(word_end_expanded, seg_end)
        overlap = max(0, overlap_end - overlap_start)

        if overlap > 0:
            overlaps.append((overlap, speaker_id))

    if not overlaps:
        # No overlap found
        return None if strict_sad else "UNKNOWN"

    # Sort by overlap (descending)
    overlaps.sort(reverse=True, key=lambda x: x[0])

    # Check for ties
    max_overlap = overlaps[0][0]
    candidates = [spk for ovl, spk in overlaps if ovl == max_overlap]

    if len(candidates) > 1:
        # Tie: multiple speakers with equal overlap
        return "UNK"

    return candidates[0]


def align_words_to_speakers(
    words: List[Dict[str, Any]],
    speaker_segments: List[Tuple[float, float, str]],
    clip_name: str,
    asr_tool: str,
    dia_tool: str,
    epsilon: float = 0.05,
    strict_sad: bool = False,
) -> List[Dict[str, Any]]:
    """Align words to speakers.

    Args:
        words: List of word dictionaries from ASR
        speaker_segments: List of (start, end, speaker_id) tuples from diarization
        clip_name: Name of the audio clip
        asr_tool: Name of ASR tool used
        dia_tool: Name of diarization tool used
        epsilon: Tolerance for interval boundaries (seconds)
        strict_sad: If True, skip words outside speech segments

    Returns:
        List of aligned word records (JSONL format)
    """
    aligned_records = []
    skipped_count = 0

    for word_dict in words:
        word_text = word_dict["word"]
        word_start = word_dict["start"]
        word_end = word_dict["end"]
        word_prob = word_dict.get("probability", 1.0)

        # Find speaker
        speaker = find_speaker_for_word(
            word_start, word_end, speaker_segments, epsilon=epsilon, strict_sad=strict_sad
        )

        # Skip if strict_sad and word is outside speech
        if strict_sad and speaker is None:
            skipped_count += 1
            continue

        # Create aligned record
        record = {
            "file": clip_name,
            "tool_asr": asr_tool,
            "tool_dia": dia_tool,
            "w": word_text,
            "start": word_start,
            "end": word_end,
            "speaker": speaker if speaker else "UNKNOWN",
            "prob": word_prob,
        }

        aligned_records.append(record)

    logger.info(
        f"Aligned {len(aligned_records)} words for {clip_name} " f"(asr={asr_tool}, dia={dia_tool})"
    )

    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} words outside speech segments (strict_sad=True)")

    return aligned_records


def align_from_files(
    asr_path: str | Path,
    rttm_path: str | Path,
    output_path: str | Path,
    asr_tool: str = "asr",
    dia_tool: str = "dia",
    epsilon: float = 0.05,
    strict_sad: bool = False,
) -> List[Dict[str, Any]]:
    """Align ASR and diarization from files.

    Args:
        asr_path: Path to ASR JSON file
        rttm_path: Path to RTTM file
        output_path: Path to save aligned JSONL
        asr_tool: Name of ASR tool
        dia_tool: Name of diarization tool
        epsilon: Tolerance for interval boundaries (seconds)
        strict_sad: If True, skip words outside speech segments

    Returns:
        List of aligned records
    """
    asr_path = Path(asr_path)
    rttm_path = Path(rttm_path)
    output_path = Path(output_path)

    # Load data
    logger.info(f"Loading ASR from {asr_path}")
    words = load_asr_words(asr_path)

    logger.info(f"Loading RTTM from {rttm_path}")
    speaker_segments = load_rttm_segments(rttm_path)

    # Get clip name
    clip_name = asr_path.stem

    # Align
    aligned_records = align_words_to_speakers(
        words,
        speaker_segments,
        clip_name,
        asr_tool,
        dia_tool,
        epsilon=epsilon,
        strict_sad=strict_sad,
    )

    # Save
    from eval.utils import save_jsonl

    save_jsonl(aligned_records, output_path)

    logger.info(f"Saved aligned records to {output_path}")

    return aligned_records


def compute_cpwer_text(aligned_records: List[Dict[str, Any]]) -> str:
    """Generate concatenated text grouped by speaker for cpWER.

    Args:
        aligned_records: List of aligned word records

    Returns:
        Text with speaker labels for cpWER calculation
    """
    # Group words by speaker in temporal order
    speaker_groups = []
    current_speaker = None
    current_words = []

    for record in sorted(aligned_records, key=lambda r: r["start"]):
        speaker = record["speaker"]
        word = record["w"]

        if speaker != current_speaker:
            if current_words:
                speaker_groups.append((current_speaker, " ".join(current_words)))
            current_speaker = speaker
            current_words = [word]
        else:
            current_words.append(word)

    # Add last group
    if current_words:
        speaker_groups.append((current_speaker, " ".join(current_words)))

    # Format: [SPEAKER_ID] text
    cpwer_lines = [f"[{spk}] {text}" for spk, text in speaker_groups]
    return "\n".join(cpwer_lines)


def main():
    """Command-line interface for alignment."""
    parser = argparse.ArgumentParser(description="Align ASR words to diarization speakers")
    parser.add_argument("--asr", required=True, help="Path to ASR JSON file")
    parser.add_argument("--rttm", required=True, help="Path to RTTM file")
    parser.add_argument("--out", required=True, help="Path to output JSONL file")
    parser.add_argument("--asr-tool", default="asr", help="Name of ASR tool (for metadata)")
    parser.add_argument("--dia-tool", default="dia", help="Name of diarization tool (for metadata)")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Tolerance for interval boundaries in seconds (default: 0.05)",
    )
    parser.add_argument(
        "--strict-sad", action="store_true", help="Skip words outside speech segments"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run alignment
    align_from_files(
        args.asr,
        args.rttm,
        args.out,
        args.asr_tool,
        args.dia_tool,
        epsilon=args.epsilon,
        strict_sad=args.strict_sad,
    )


if __name__ == "__main__":
    main()
