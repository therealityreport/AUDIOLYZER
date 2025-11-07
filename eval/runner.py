#!/usr/bin/env python3
"""Main runner for ASR and diarization evaluation harness."""

import argparse
import csv
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval.align import align_from_files
from eval.asr import run_faster_whisper, run_sherpa_onnx
from eval.diar import run_pyannote, run_speechbrain
from eval.score import (
    batch_compute_der,
    batch_compute_wer,
    compute_cpwer,
    compute_wer_from_asr,
)
from eval.utils import (
    RuntimeLogger,
    ensure_dir,
    get_audio_duration,
    get_audio_files,
    load_config,
    time_operation,
)

logger = logging.getLogger(__name__)


class EvalRunner:
    """Evaluation harness runner."""

    def __init__(self, config_path: str | Path):
        """Initialize runner.

        Args:
            config_path: Path to eval.yaml config file
        """
        self.config = load_config(config_path)
        self.config_path = Path(config_path)

        # Setup paths
        self.input_dir = Path(self.config["paths"]["input_dir"])
        self.output_dir = Path(self.config["paths"]["output_dir"])

        # Create output directories
        self.wav_dir = ensure_dir(self.output_dir / "wavs")
        self.asr_dir = ensure_dir(self.output_dir / "asr")
        self.dia_dir = ensure_dir(self.output_dir / "dia")
        self.aligned_dir = ensure_dir(self.output_dir / "aligned")
        self.scores_dir = ensure_dir(self.output_dir / "scores")
        self.logs_dir = ensure_dir(self.output_dir / "logs")

        # Runtime logger
        self.runtime_logger = RuntimeLogger(self.logs_dir / "runtime.jsonl")

        # Get list of clips to process
        self.clips = self._get_clips()

        logger.info(f"Initialized eval runner with {len(self.clips)} clips")

    def _get_clips(self) -> List[Path]:
        """Get list of clips to process.

        Returns:
            List of input file paths
        """
        # If clips specified in config, use those
        clip_names = self.config.get("clips", [])

        if clip_names:
            clips = [self.input_dir / name for name in clip_names]
            # Verify they exist
            clips = [c for c in clips if c.exists()]
        else:
            # Get all audio/video files in input directory
            clips = get_audio_files(self.input_dir)

        return clips

    def stage1_extract_wavs(self) -> Dict[str, Path]:
        """Stage 1: Extract mono 16kHz WAV from input files.

        Returns:
            Dictionary mapping clip stems to WAV paths
        """
        logger.info("=" * 60)
        logger.info("STAGE 1: Extract WAVs")
        logger.info("=" * 60)

        wav_files = {}

        for clip_path in self.clips:
            clip_stem = clip_path.stem
            wav_path = self.wav_dir / f"{clip_stem}.wav"

            # Skip if already exists
            if wav_path.exists():
                logger.info(f"WAV already exists: {wav_path.name}")
                wav_files[clip_stem] = wav_path
                continue

            logger.info(f"Extracting WAV from {clip_path.name}")

            try:
                with time_operation(f"Extract {clip_path.name}") as timer:
                    # Use ffmpeg to extract mono 16kHz WAV
                    cmd = [
                        "ffmpeg",
                        "-i",
                        str(clip_path),
                        "-ac",
                        "1",  # mono
                        "-ar",
                        "16000",  # 16kHz
                        "-acodec",
                        "pcm_s16le",  # 16-bit PCM
                        "-y",  # overwrite
                        str(wav_path),
                    ]

                    subprocess.run(cmd, check=True, capture_output=True)

                wav_files[clip_stem] = wav_path

                # Log runtime
                self.runtime_logger.log_run(
                    clip=clip_stem, tool="ffmpeg", stage="extract", duration=timer.elapsed
                )

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to extract WAV from {clip_path.name}: {e}")
                logger.error(e.stderr.decode() if e.stderr else "")

        logger.info(f"Extracted {len(wav_files)} WAV files")
        return wav_files

    def stage2_run_asr(self, wav_files: Dict[str, Path]) -> Dict[str, Dict[str, Path]]:
        """Stage 2: Run ASR tools.

        Args:
            wav_files: Dictionary mapping clip stems to WAV paths

        Returns:
            Nested dict: {tool_name: {clip_stem: asr_json_path}}
        """
        logger.info("=" * 60)
        logger.info("STAGE 2: Run ASR")
        logger.info("=" * 60)

        asr_results = {}

        # Get enabled ASR tools
        asr_config = self.config.get("asr", {})

        # faster-whisper
        if asr_config.get("faster_whisper", {}).get("enabled", False):
            logger.info("\nRunning faster-whisper...")
            fw_results = self._run_asr_tool(wav_files, "faster_whisper", run_faster_whisper)
            asr_results["faster_whisper"] = fw_results

        # sherpa-onnx
        if asr_config.get("sherpa_onnx", {}).get("enabled", False):
            logger.info("\nRunning sherpa-onnx...")
            so_results = self._run_asr_tool(wav_files, "sherpa_onnx", run_sherpa_onnx)
            asr_results["sherpa_onnx"] = so_results

        return asr_results

    def _run_asr_tool(
        self, wav_files: Dict[str, Path], tool_name: str, run_func
    ) -> Dict[str, Path]:
        """Run a single ASR tool on all clips.

        Args:
            wav_files: Dictionary mapping clip stems to WAV paths
            tool_name: Name of the tool
            run_func: Function to run the tool

        Returns:
            Dictionary mapping clip stems to output JSON paths
        """
        tool_dir = ensure_dir(self.asr_dir / tool_name)
        results = {}

        for clip_stem, wav_path in wav_files.items():
            output_path = tool_dir / f"{clip_stem}.json"

            # Skip if already exists
            if output_path.exists():
                logger.info(f"ASR output exists: {output_path.name}")
                results[clip_stem] = output_path
                continue

            logger.info(f"Processing {clip_stem} with {tool_name}")

            try:
                with time_operation(f"{tool_name} on {clip_stem}") as timer:
                    run_func(wav_path, output_path, self.config)

                results[clip_stem] = output_path

                # Log runtime with RTF
                audio_duration = get_audio_duration(wav_path)
                self.runtime_logger.log_run(
                    clip=clip_stem,
                    tool=tool_name,
                    stage="asr",
                    duration=timer.elapsed,
                    audio_duration=audio_duration,
                )

            except Exception as e:
                logger.error(f"ASR failed for {clip_stem} with {tool_name}: {e}")

        return results

    def stage3_run_diarization(self, wav_files: Dict[str, Path]) -> Dict[str, Dict[str, Path]]:
        """Stage 3: Run diarization tools.

        Args:
            wav_files: Dictionary mapping clip stems to WAV paths

        Returns:
            Nested dict: {tool_name: {clip_stem: rttm_path}}
        """
        logger.info("=" * 60)
        logger.info("STAGE 3: Run Diarization")
        logger.info("=" * 60)

        dia_results = {}

        # Get enabled diarization tools
        dia_config = self.config.get("diarization", {})

        # pyannote
        if dia_config.get("pyannote", {}).get("enabled", False):
            logger.info("\nRunning pyannote...")
            pya_results = self._run_dia_tool(wav_files, "pyannote", run_pyannote)
            dia_results["pyannote"] = pya_results

        # speechbrain
        if dia_config.get("speechbrain", {}).get("enabled", False):
            logger.info("\nRunning SpeechBrain...")
            sb_results = self._run_dia_tool(wav_files, "speechbrain", run_speechbrain)
            dia_results["speechbrain"] = sb_results

        return dia_results

    def _run_dia_tool(
        self, wav_files: Dict[str, Path], tool_name: str, run_func
    ) -> Dict[str, Path]:
        """Run a single diarization tool on all clips.

        Args:
            wav_files: Dictionary mapping clip stems to WAV paths
            tool_name: Name of the tool
            run_func: Function to run the tool

        Returns:
            Dictionary mapping clip stems to output RTTM paths
        """
        tool_dir = ensure_dir(self.dia_dir / tool_name)
        results = {}

        for clip_stem, wav_path in wav_files.items():
            output_path = tool_dir / f"{clip_stem}.rttm"

            # Skip if already exists
            if output_path.exists():
                logger.info(f"RTTM output exists: {output_path.name}")
                results[clip_stem] = output_path
                continue

            logger.info(f"Diarizing {clip_stem} with {tool_name}")

            try:
                with time_operation(f"{tool_name} on {clip_stem}") as timer:
                    run_func(wav_path, output_path, self.config)

                results[clip_stem] = output_path

                # Log runtime with RTF
                audio_duration = get_audio_duration(wav_path)
                self.runtime_logger.log_run(
                    clip=clip_stem,
                    tool=tool_name,
                    stage="diarization",
                    duration=timer.elapsed,
                    audio_duration=audio_duration,
                )

            except Exception as e:
                logger.error(f"Diarization failed for {clip_stem} with {tool_name}: {e}")

        return results

    def stage4_align(
        self, asr_results: Dict[str, Dict[str, Path]], dia_results: Dict[str, Dict[str, Path]]
    ) -> Dict[str, Dict[str, Path]]:
        """Stage 4: Align ASR words to diarization speakers.

        Args:
            asr_results: ASR results from stage 2
            dia_results: Diarization results from stage 3

        Returns:
            Nested dict: {(asr_tool, dia_tool): {clip_stem: aligned_path}}
        """
        logger.info("=" * 60)
        logger.info("STAGE 4: Align Words to Speakers")
        logger.info("=" * 60)

        aligned_results = {}

        # For each combination of ASR and diarization tool
        for asr_tool, asr_clips in asr_results.items():
            for dia_tool, dia_clips in dia_results.items():
                combo_key = f"{asr_tool}_{dia_tool}"
                logger.info(f"\nAligning {asr_tool} + {dia_tool}")

                combo_results = {}

                for clip_stem in asr_clips:
                    if clip_stem not in dia_clips:
                        continue

                    asr_path = asr_clips[clip_stem]
                    rttm_path = dia_clips[clip_stem]

                    output_path = self.aligned_dir / f"{clip_stem}_{combo_key}.jsonl"

                    # Skip if exists
                    if output_path.exists():
                        logger.info(f"Alignment exists: {output_path.name}")
                        combo_results[clip_stem] = output_path
                        continue

                    logger.info(f"Aligning {clip_stem}")

                    try:
                        with time_operation(f"Align {clip_stem}") as timer:
                            align_from_files(
                                asr_path,
                                rttm_path,
                                output_path,
                                asr_tool=asr_tool,
                                dia_tool=dia_tool,
                            )

                        combo_results[clip_stem] = output_path

                        self.runtime_logger.log_run(
                            clip=clip_stem,
                            tool=combo_key,
                            stage="alignment",
                            duration=timer.elapsed,
                        )

                    except Exception as e:
                        logger.error(f"Alignment failed for {clip_stem}: {e}")

                aligned_results[combo_key] = combo_results

        return aligned_results

    def stage5_score(
        self,
        wav_files: Dict[str, Path],
        asr_results: Dict[str, Dict[str, Path]],
        dia_results: Dict[str, Dict[str, Path]],
        aligned_results: Dict[str, Dict[str, Path]],
    ) -> None:
        """Stage 5: Compute scores and generate summary CSV.

        Args:
            wav_files: WAV file paths
            asr_results: ASR results
            dia_results: Diarization results
            aligned_results: Alignment results
        """
        logger.info("=" * 60)
        logger.info("STAGE 5: Score and Generate Summary")
        logger.info("=" * 60)

        summary_rows = []

        # Get reference paths if available
        ref_config = self.config.get("references", {})
        ref_transcripts_dir = ref_config.get("transcripts_dir")
        ref_rttm_dir = ref_config.get("rttm_dir")

        # For each clip and tool combination
        for clip_stem, wav_path in wav_files.items():
            audio_duration = get_audio_duration(wav_path)

            # For each ASR tool
            for asr_tool, asr_clips in asr_results.items():
                if clip_stem not in asr_clips:
                    continue

                asr_path = asr_clips[clip_stem]

                # For each diarization tool
                for dia_tool, dia_clips in dia_results.items():
                    if clip_stem not in dia_clips:
                        continue

                    rttm_path = dia_clips[clip_stem]
                    combo_key = f"{asr_tool}_{dia_tool}"

                    logger.info(f"\nScoring {clip_stem} with {combo_key}")

                    row = {
                        "clip": clip_stem,
                        "asr": asr_tool,
                        "dia": dia_tool,
                        "duration": round(audio_duration, 2),
                    }

                    # Compute WER if reference available
                    if ref_transcripts_dir:
                        ref_txt = Path(ref_transcripts_dir) / f"{clip_stem}.txt"
                        if ref_txt.exists():
                            try:
                                wer_metrics = compute_wer_from_asr(asr_path, reference_file=ref_txt)
                                row["wer"] = wer_metrics["wer_percentage"]
                            except Exception as e:
                                logger.error(f"WER computation failed: {e}")
                                row["wer"] = "N/A"
                        else:
                            row["wer"] = "N/A"
                    else:
                        row["wer"] = "N/A"

                    # cpWER (placeholder - needs speaker-separated reference)
                    row["cpwer"] = "N/A"

                    # Compute DER if reference available
                    if ref_rttm_dir:
                        ref_rttm = Path(ref_rttm_dir) / f"{clip_stem}.rttm"
                        if ref_rttm.exists():
                            try:
                                from eval.score import compute_diarization_metrics

                                der_metrics = compute_diarization_metrics(
                                    ref_rttm, rttm_path, total_duration=audio_duration
                                )
                                row["der"] = der_metrics.get("DER", "N/A")
                                row["jer"] = der_metrics.get("JER", "N/A")
                            except Exception as e:
                                logger.error(f"DER computation failed: {e}")
                                row["der"] = "N/A"
                                row["jer"] = "N/A"
                        else:
                            row["der"] = "N/A"
                            row["jer"] = "N/A"
                    else:
                        row["der"] = "N/A"
                        row["jer"] = "N/A"

                    # Get RTF from runtime log
                    asr_stats = self.runtime_logger.get_stats(tool=asr_tool, stage="asr")
                    dia_stats = self.runtime_logger.get_stats(tool=dia_tool, stage="diarization")

                    row["rtf_asr"] = round(asr_stats.get("mean_rtf", 0), 4)
                    row["rtf_dia"] = round(dia_stats.get("mean_rtf", 0), 4)

                    row["notes"] = ""

                    summary_rows.append(row)

        # Write summary CSV
        summary_path = self.scores_dir / "summary.csv"
        self._write_summary_csv(summary_rows, summary_path)

        # Save runtime log
        self.runtime_logger.save()

        logger.info(f"\nSummary written to: {summary_path}")
        logger.info(f"Runtime log written to: {self.runtime_logger.log_file}")

    def _write_summary_csv(self, rows: List[Dict[str, Any]], output_path: Path) -> None:
        """Write summary CSV.

        Args:
            rows: List of row dictionaries
            output_path: Output CSV path
        """
        if not rows:
            logger.warning("No rows to write to summary CSV")
            return

        fieldnames = [
            "clip",
            "asr",
            "dia",
            "duration",
            "wer",
            "cpwer",
            "der",
            "jer",
            "rtf_asr",
            "rtf_dia",
            "notes",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Wrote {len(rows)} rows to {output_path}")

    def run_all_stages(self, skip_stages: set = None) -> None:
        """Run all evaluation stages.

        Args:
            skip_stages: Set of stage numbers to skip ('1', '2', '3', '4', '5')
        """
        if skip_stages is None:
            skip_stages = set()

        logger.info("Starting evaluation harness")
        logger.info(f"Config: {self.config_path}")
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Clips: {len(self.clips)}")

        if skip_stages:
            logger.info(f"Skipping stages: {', '.join(sorted(skip_stages))}")

        # Stage 1: Extract WAVs
        if "1" not in skip_stages:
            wav_files = self.stage1_extract_wavs()
        else:
            logger.info("Skipping stage 1 (extract WAVs)")
            wav_files = {p.stem: p for p in self.wav_dir.glob("*.wav")}

        # Stage 2: Run ASR
        if "2" not in skip_stages:
            asr_results = self.stage2_run_asr(wav_files)
        else:
            logger.info("Skipping stage 2 (ASR)")
            asr_results = {}
            for tool_dir in self.asr_dir.iterdir():
                if tool_dir.is_dir():
                    asr_results[tool_dir.name] = {p.stem: p for p in tool_dir.glob("*.json")}

        # Stage 3: Run diarization
        if "3" not in skip_stages:
            dia_results = self.stage3_run_diarization(wav_files)
        else:
            logger.info("Skipping stage 3 (diarization)")
            dia_results = {}
            for tool_dir in self.dia_dir.iterdir():
                if tool_dir.is_dir():
                    dia_results[tool_dir.name] = {p.stem: p for p in tool_dir.glob("*.rttm")}

        # Stage 4: Align
        if "4" not in skip_stages:
            aligned_results = self.stage4_align(asr_results, dia_results)
        else:
            logger.info("Skipping stage 4 (alignment)")
            aligned_results = {}
            for aligned_file in self.aligned_dir.glob("*.jsonl"):
                parts = aligned_file.stem.rsplit("_", 2)
                if len(parts) == 3:
                    clip, asr, dia = parts
                    combo_key = f"{asr}_{dia}"
                    if combo_key not in aligned_results:
                        aligned_results[combo_key] = {}
                    aligned_results[combo_key][clip] = aligned_file

        # Stage 5: Score
        if "5" not in skip_stages:
            self.stage5_score(wav_files, asr_results, dia_results, aligned_results)
        else:
            logger.info("Skipping stage 5 (scoring)")

        logger.info("\n" + "=" * 60)
        logger.info("Evaluation complete!")
        logger.info("=" * 60)


def print_version_banner():
    """Print version banner with tool information."""
    from eval import __version__
    from eval.utils import get_python_version, get_platform_info, get_package_versions

    # Get git info
    git_commit = "unknown"
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        git_commit = result.stdout.strip()
    except Exception:
        pass

    print("=" * 70)
    print(f"ASR & Diarization Evaluation Harness v{__version__}")
    print(f"Git: {git_commit} | Python: {get_python_version()} | {get_platform_info()['system']}")
    print("=" * 70)

    # Print tool versions
    packages = get_package_versions()
    tools = {
        "faster-whisper": packages.get("faster-whisper", "not installed"),
        "sherpa-onnx": packages.get("sherpa-onnx", "not installed"),
        "pyannote.audio": packages.get("pyannote.audio", "not installed"),
        "speechbrain": packages.get("speechbrain", "not installed"),
        "torch": packages.get("torch", "not installed"),
    }

    print("Tool Versions:")
    for tool, version in tools.items():
        status = "✓" if version != "not installed" else "✗"
        print(f"  {status} {tool:20} {version}")
    print("=" * 70)
    print()


def parse_pairs(pairs_str: str) -> tuple:
    """Parse --pairs argument.

    Format: "asr=fw,sherpa dia=pyannote,speechbrain"

    Returns:
        Tuple of (asr_tools, dia_tools) lists
    """
    asr_tools = []
    dia_tools = []

    parts = pairs_str.split()
    for part in parts:
        if "=" not in part:
            continue

        key, value = part.split("=", 1)
        tools = [t.strip() for t in value.split(",")]

        if key == "asr":
            asr_tools = tools
        elif key == "dia":
            dia_tools = tools

    return asr_tools, dia_tools


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ASR and Diarization Evaluation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all stages
  %(prog)s --config eval/eval.yaml

  # Run specific tool pairs
  %(prog)s --pairs "asr=faster_whisper,sherpa_onnx dia=pyannote,speechbrain"

  # Skip stages
  %(prog)s --skip extract,score

  # Clean old outputs
  %(prog)s --clean
        """,
    )
    parser.add_argument(
        "--config",
        default="eval/eval.yaml",
        help="Path to configuration file (default: eval/eval.yaml)",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "1", "2", "3", "4", "5"],
        default="all",
        help="Stage to run: all, 1 (extract), 2 (asr), 3 (dia), 4 (align), 5 (score)",
    )
    parser.add_argument(
        "--pairs",
        metavar="SPEC",
        help='Tool pairs to run, e.g.: "asr=faster_whisper,sherpa_onnx dia=pyannote,speechbrain"',
    )
    parser.add_argument(
        "--skip",
        metavar="STAGES",
        help="Comma-separated stages to skip: extract,asr,dia,align,score or 1,2,3,4,5",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean outputs/tmp and logs older than 7 days before running",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Print version banner
    print_version_banner()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Handle cleanup
    if args.clean:
        from eval.utils import ensure_dir
        import time

        logger.info("Cleaning old outputs...")
        output_dir = Path("outputs")

        # Clean tmp directory
        tmp_dir = output_dir / "tmp"
        if tmp_dir.exists():
            import shutil

            shutil.rmtree(tmp_dir)
            logger.info(f"Removed {tmp_dir}")

        # Clean logs older than 7 days
        logs_dir = output_dir / "logs"
        if logs_dir.exists():
            cutoff_time = time.time() - (7 * 24 * 60 * 60)  # 7 days
            for log_file in logs_dir.glob("*"):
                if log_file.is_file() and log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    logger.info(f"Removed old log: {log_file.name}")

    # Create runner
    runner = EvalRunner(args.config)

    # Save environment manifest
    from eval.utils import save_env_manifest

    save_env_manifest(runner.logs_dir / "env.json")

    # Parse tool pairs if specified
    if args.pairs:
        asr_tools, dia_tools = parse_pairs(args.pairs)
        logger.info(f"Running specific pairs: ASR={asr_tools}, DIA={dia_tools}")

        # Override config
        for asr_tool in runner.config.get("asr", {}).keys():
            runner.config["asr"][asr_tool]["enabled"] = asr_tool in asr_tools

        for dia_tool in runner.config.get("diarization", {}).keys():
            runner.config["diarization"][dia_tool]["enabled"] = dia_tool in dia_tools

    # Parse skip stages
    skip_stages = set()
    if args.skip:
        skip_parts = [s.strip() for s in args.skip.split(",")]
        for part in skip_parts:
            # Map names to numbers
            stage_map = {"extract": "1", "asr": "2", "dia": "3", "align": "4", "score": "5"}
            skip_stages.add(stage_map.get(part, part))

    # Run stages
    if args.stage == "all":
        runner.run_all_stages(skip_stages=skip_stages)
    elif args.stage == "1":
        runner.stage1_extract_wavs()
    elif args.stage == "2":
        wav_files = {p.stem: p for p in runner.wav_dir.glob("*.wav")}
        runner.stage2_run_asr(wav_files)
    elif args.stage == "3":
        wav_files = {p.stem: p for p in runner.wav_dir.glob("*.wav")}
        runner.stage3_run_diarization(wav_files)
    elif args.stage == "4":
        # Load existing results
        asr_results = {}
        dia_results = {}
        for tool_dir in runner.asr_dir.iterdir():
            if tool_dir.is_dir():
                asr_results[tool_dir.name] = {p.stem: p for p in tool_dir.glob("*.json")}
        for tool_dir in runner.dia_dir.iterdir():
            if tool_dir.is_dir():
                dia_results[tool_dir.name] = {p.stem: p for p in tool_dir.glob("*.rttm")}
        runner.stage4_align(asr_results, dia_results)
    elif args.stage == "5":
        # Load existing results
        wav_files = {p.stem: p for p in runner.wav_dir.glob("*.wav")}
        asr_results = {}
        dia_results = {}
        aligned_results = {}

        for tool_dir in runner.asr_dir.iterdir():
            if tool_dir.is_dir():
                asr_results[tool_dir.name] = {p.stem: p for p in tool_dir.glob("*.json")}

        for tool_dir in runner.dia_dir.iterdir():
            if tool_dir.is_dir():
                dia_results[tool_dir.name] = {p.stem: p for p in tool_dir.glob("*.rttm")}

        for aligned_file in runner.aligned_dir.glob("*.jsonl"):
            # Parse filename: {clip}_{asr}_{dia}.jsonl
            parts = aligned_file.stem.rsplit("_", 2)
            if len(parts) == 3:
                clip, asr, dia = parts
                combo_key = f"{asr}_{dia}"
                if combo_key not in aligned_results:
                    aligned_results[combo_key] = {}
                aligned_results[combo_key][clip] = aligned_file

        runner.stage5_score(wav_files, asr_results, dia_results, aligned_results)


if __name__ == "__main__":
    main()
