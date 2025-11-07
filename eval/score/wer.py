"""WER (Word Error Rate) scoring."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def levenshtein_distance(ref: List[str], hyp: List[str]) -> tuple:
    """Calculate Levenshtein distance between two sequences.

    Args:
        ref: Reference word list
        hyp: Hypothesis word list

    Returns:
        Tuple of (distance, substitutions, insertions, deletions)
    """
    m, n = len(ref), len(hyp)

    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Track operations
    ops = [[None] * (n + 1) for _ in range(m + 1)]

    # Initialize
    for i in range(m + 1):
        dp[i][0] = i
        ops[i][0] = "D"  # deletion

    for j in range(n + 1):
        dp[0][j] = j
        ops[0][j] = "I"  # insertion

    ops[0][0] = "C"  # correct

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                ops[i][j] = "C"  # correct
            else:
                # Substitution
                sub = dp[i - 1][j - 1] + 1
                # Deletion
                delete = dp[i - 1][j] + 1
                # Insertion
                insert = dp[i][j - 1] + 1

                min_cost = min(sub, delete, insert)
                dp[i][j] = min_cost

                if min_cost == sub:
                    ops[i][j] = "S"  # substitution
                elif min_cost == delete:
                    ops[i][j] = "D"  # deletion
                else:
                    ops[i][j] = "I"  # insertion

    # Backtrack to count operations
    i, j = m, n
    subs, ins, dels = 0, 0, 0

    while i > 0 or j > 0:
        op = ops[i][j]

        if op == "C":
            i -= 1
            j -= 1
        elif op == "S":
            subs += 1
            i -= 1
            j -= 1
        elif op == "D":
            dels += 1
            i -= 1
        elif op == "I":
            ins += 1
            j -= 1

    return dp[m][n], subs, ins, dels


def normalize_text(text: str) -> str:
    """Normalize text for WER calculation.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    import re

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Normalize whitespace
    text = " ".join(text.split())

    return text


def compute_wer(reference: str, hypothesis: str) -> Dict[str, Any]:
    """Compute Word Error Rate.

    WER = (S + I + D) / N
    where:
        S = substitutions
        I = insertions
        D = deletions
        N = number of words in reference

    Args:
        reference: Reference text
        hypothesis: Hypothesis text

    Returns:
        Dictionary with WER metrics
    """
    # Normalize
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    # Split into words
    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    # Handle empty cases
    if not ref_words:
        return {
            "wer": 0.0 if not hyp_words else float("inf"),
            "substitutions": 0,
            "insertions": len(hyp_words),
            "deletions": 0,
            "num_words": 0,
            "num_errors": len(hyp_words),
        }

    # Compute Levenshtein distance
    distance, subs, ins, dels = levenshtein_distance(ref_words, hyp_words)

    # Calculate WER
    num_words = len(ref_words)
    wer = (subs + ins + dels) / num_words if num_words > 0 else 0.0

    return {
        "wer": round(wer, 4),
        "wer_percentage": round(wer * 100, 2),
        "substitutions": subs,
        "insertions": ins,
        "deletions": dels,
        "num_words": num_words,
        "num_errors": subs + ins + dels,
    }


def compute_wer_from_asr(
    asr_json_path: str | Path,
    reference_text: Optional[str] = None,
    reference_file: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Compute WER from ASR JSON output.

    Args:
        asr_json_path: Path to ASR JSON file
        reference_text: Reference text (if provided)
        reference_file: Path to reference text file (if provided)

    Returns:
        WER metrics
    """
    from eval.utils import load_json

    if reference_text is None and reference_file is None:
        raise ValueError("Either reference_text or reference_file must be provided")

    # Load reference
    if reference_file:
        with open(reference_file, "r", encoding="utf-8") as f:
            reference_text = f.read()

    # Load ASR output
    asr_data = load_json(asr_json_path)

    # Extract hypothesis text
    hypothesis_text = ""
    for segment in asr_data.get("segments", []):
        hypothesis_text += segment.get("text", "") + " "

    hypothesis_text = hypothesis_text.strip()

    # Compute WER
    return compute_wer(reference_text, hypothesis_text)


def compute_cpwer(
    aligned_records: List[Dict[str, Any]], reference_by_speaker: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Compute conversation-preserving WER (cpWER).

    cpWER groups words by speaker before computing WER.

    Args:
        aligned_records: List of aligned word records
        reference_by_speaker: Optional reference text grouped by speaker

    Returns:
        cpWER metrics
    """
    # Group hypothesis words by speaker
    speaker_words = {}

    for record in sorted(aligned_records, key=lambda r: r["start"]):
        speaker = record["speaker"]
        word = record["w"]

        if speaker not in speaker_words:
            speaker_words[speaker] = []

        speaker_words[speaker].append(word)

    # Build hypothesis text with speaker labels
    hypothesis_lines = []
    for speaker, words in sorted(speaker_words.items()):
        hypothesis_lines.append(f"[{speaker}] {' '.join(words)}")

    hypothesis_text = "\n".join(hypothesis_lines)

    # If we have reference text, compute WER
    if reference_by_speaker:
        reference_lines = []
        for speaker, text in sorted(reference_by_speaker.items()):
            reference_lines.append(f"[{speaker}] {text}")

        reference_text = "\n".join(reference_lines)

        return compute_wer(reference_text, hypothesis_text)

    # Otherwise, just return the structured hypothesis
    return {
        "hypothesis": hypothesis_text,
        "speaker_count": len(speaker_words),
        "note": "No reference provided for cpWER calculation",
    }


def batch_compute_wer(
    asr_results: Dict[str, str | Path], reference_files: Dict[str, str | Path]
) -> Dict[str, Dict[str, Any]]:
    """Compute WER for multiple ASR results.

    Args:
        asr_results: Dictionary mapping clip names to ASR JSON paths
        reference_files: Dictionary mapping clip names to reference text files

    Returns:
        Dictionary mapping clip names to WER metrics
    """
    results = {}

    for clip_name in asr_results:
        asr_path = asr_results[clip_name]

        if clip_name not in reference_files:
            logger.warning(f"No reference found for {clip_name}, skipping WER")
            continue

        ref_path = reference_files[clip_name]

        try:
            wer_metrics = compute_wer_from_asr(asr_path, reference_file=ref_path)
            results[clip_name] = wer_metrics
            logger.info(f"{clip_name}: WER = {wer_metrics['wer_percentage']:.2f}%")

        except Exception as e:
            logger.error(f"Failed to compute WER for {clip_name}: {e}")
            results[clip_name] = {"error": str(e)}

    return results
