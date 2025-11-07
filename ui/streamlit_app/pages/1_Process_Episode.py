"""Episode ingestion and audio preprocessing workflow."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_PACKAGE_ROOT = _Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _PACKAGE_ROOT.parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
for _candidate in (_PROJECT_ROOT, _SRC_DIR):
    candidate_str = str(_candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import re
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import streamlit as st

try:  # Optional dependency for quality mode indicator
    import clearvoice  # type: ignore  # noqa: F401

    _CLEARERVOICE_AVAILABLE = True
except ImportError:  # pragma: no cover - runtime check
    _CLEARERVOICE_AVAILABLE = False

from pipeline.audio_ops import AudioPreprocessingError  # noqa: E402
from ui.streamlit_app.bootstrap import initialise_paths

ROOT, SRC_DIR = initialise_paths()
from show_scribe.exceptions import AudioPreprocessingCancelled  # noqa: E402
from show_scribe.ui_ops import (  # noqa: E402
    cancel_active_audio_runs,
    create_audio,
    ensure_episode_scaffold,
    read_metadata,
    resolve_artifact_path,
    update_metadata,
)
from ui.streamlit_app.components.artifact_badge import render_artifact_badge  # noqa: E402
from ui.streamlit_app.components.episode_picker import (  # noqa: E402
    SESSION_DIR_KEY,
    SESSION_LABEL_KEY,
)
from ui.streamlit_app.components.layout import render_global_sidebar  # noqa: E402
from ui.streamlit_app.components.settings_manager import SettingsManager  # noqa: E402

VIDEO_LIBRARY_DIR = ROOT / "data" / "video-files"
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v")
st.set_page_config(page_title="Process Episode", layout="wide")
active_label, active_dir, active_metadata = render_global_sidebar()
settings = SettingsManager.load_settings()
if active_dir:
    active_metadata = read_metadata(active_dir) or active_metadata or {}
else:
    active_metadata = active_metadata or {}
DEFAULTS: dict[str, Any] = {
    "process_show_slug": "",
    "process_episode_id": "",
    "process_show_config": settings.get("default_show_config") or "",
    "process_preset_path": settings.get("default_preset") or "configs/balanced.yaml",
    "process_enable_separation": True,
    "process_enable_enhancement": True,
    "process_enhancer_provider": "resemble",
    "process_sample_rate": 16000,
    "process_channels": 1,
    "process_target_lufs": -20.0,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)

# Migrate old default preset to new balanced.yaml default
if st.session_state.get("process_preset_path") == "configs/reality_tv.yaml":
    st.session_state["process_preset_path"] = "configs/balanced.yaml"

flash = st.session_state.pop("process_flash", None)
if flash:
    variant = flash.get("variant", "success")
    message = flash.get("message", "Operation completed.")
    if variant == "warning":
        st.warning(message)
    elif variant == "info":
        st.info(message)
    elif variant == "error":
        st.error(message)
    else:
        st.success(message)
    artifacts = flash.get("artifacts")
    if isinstance(artifacts, dict) and active_dir:
        st.caption("Created artifacts:")
        for label, path in artifacts.items():
            friendly = label.replace("_", " ").title()
            st.write(f"- **{friendly}**: `{path}`")
st.title("Process Episode")
st.write(
    "Configure the episode identifiers, choose a source video, and run audio preprocessing. "
    "Outputs are written under the configured shows root using the canonical "
    "`shows/<show_slug>/episodes/<episode_id>/` layout."
)
previous_log = st.session_state.get("process_last_log")
if isinstance(previous_log, dict) and previous_log.get("episode_dir"):
    with st.expander("Last CREATE AUDIO run", expanded=False):
        st.write(f"Episode directory: `{previous_log['episode_dir']}`")
        if previous_log.get("error"):
            st.error(previous_log["error"])
        artifacts_log = previous_log.get("artifacts")
        if artifacts_log:
            st.json(artifacts_log)
        messages_log = previous_log.get("messages")
        if isinstance(messages_log, list) and messages_log:
            st.caption("Progress log:")
            for entry in messages_log:
                message_text = str(entry.get("message", ""))
                level = str(entry.get("level", "info")).lower()
                percent = entry.get("percent")
                timestamp = entry.get("timestamp")
                if isinstance(percent, (int, float)):
                    percent_label = f"{int(percent):3d}%"
                else:
                    percent_label = "--%"
                level_tag = {
                    "error": "[ERR]",
                    "warning": "[WARN]",
                    "success": "[OK]",
                    "info": "[INFO]",
                }.get(level, "[INFO]")
                if isinstance(timestamp, str) and timestamp:
                    st.write(f"{level_tag} {percent_label} {message_text} ({timestamp})")
                else:
                    st.write(f"{level_tag} {percent_label} {message_text}")


def _guess_episode_from_path(video_path: str) -> dict[str, str] | None:
    pattern = re.compile(
        r"(?P<show>[A-Za-z0-9]+)[^A-Za-z0-9]+S(?P<season>\d{1,2})E(?P<episode>\d{1,2})(?P<suffix>.*)",
        re.IGNORECASE,
    )
    filename = Path(video_path).stem
    match = pattern.search(filename)
    if not match:
        return None
    show = match.group("show").upper()
    season = int(match.group("season"))
    episode = int(match.group("episode"))
    suffix = match.group("suffix").strip("_- ")
    episode_id = f"{show}_S{season:02d}E{episode:02d}"
    if suffix:
        cleaned = re.sub(r"[^A-Za-z0-9]+", "_", suffix).strip("_")
        if cleaned:
            episode_id = f"{episode_id}_{cleaned.upper()}"
    return {"show_slug": show, "episode_id": episode_id, "source": filename}


def _list_video_library() -> list[Path]:
    if not VIDEO_LIBRARY_DIR.exists():
        return []
    files: set[Path] = set()
    for ext in VIDEO_EXTS:
        files.update(path for path in VIDEO_LIBRARY_DIR.glob(f"**/*{ext}") if path.is_file())
    return sorted(files, key=lambda value: value.name.lower())


st.subheader("Episode Configuration")
config_cols = st.columns(2)
show_slug = config_cols[0].text_input("Show slug", key="process_show_slug")
episode_id = config_cols[1].text_input("Episode ID", key="process_episode_id")
show_config_path = st.text_input(
    "Show config path (optional)",
    key="process_show_config",
    help="Overrides the default show_config.json for this episode if provided.",
)
st.subheader("Video Selection")
st.caption(f"Library root: {VIDEO_LIBRARY_DIR}")

# File upload widget
uploaded_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mov", "mkv", "avi", "m4v"],
    help="Upload a video file to add it to your library. It will be saved to the video library directory.",
)
if uploaded_file is not None:
    # Ensure library directory exists
    VIDEO_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)

    # Find unique filename if file already exists
    target_path = VIDEO_LIBRARY_DIR / uploaded_file.name
    if target_path.exists():
        # Auto-rename with (1), (2), etc.
        stem = target_path.stem
        suffix = target_path.suffix
        counter = 1
        while target_path.exists():
            target_path = VIDEO_LIBRARY_DIR / f"{stem} ({counter}){suffix}"
            counter += 1
        st.info(f"File renamed to '{target_path.name}' to avoid overwriting existing file.")

    # Save uploaded file
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✓ Uploaded '{target_path.name}' to library")

    # Auto-select the uploaded file
    st.session_state["process_video_selection"] = target_path.name
    st.rerun()

library_files = _list_video_library()
selected_video: Path | None = None
if library_files:
    option_labels = []
    for entry in library_files:
        try:
            label = entry.relative_to(VIDEO_LIBRARY_DIR).as_posix()
        except ValueError:
            label = entry.name
        option_labels.append(label)
    option_labels.append("Custom path…")
    default_label = st.session_state.get("process_video_selection") or option_labels[0]
    if default_label not in option_labels:
        default_label = option_labels[0]
    selected_label = st.selectbox(
        "Source video file",
        options=option_labels,
        index=option_labels.index(default_label),
        key="process_video_selection",
    )
    if selected_label == "Custom path…":
        custom_path = st.text_input(
            "Custom source video path",
            key="process_video_custom",
            placeholder="/path/to/source_video.mp4",
        )
        selected_video = Path(custom_path).expanduser().resolve() if custom_path.strip() else None
    else:
        selected_video = library_files[option_labels.index(selected_label)].resolve()
    if selected_video:
        st.caption(str(selected_video))
else:
    st.warning(
        "No video files detected in `data/video-files/`. Place source media there or provide a custom path."
    )
    custom_path = st.text_input(
        "Source video file",
        key="process_video_custom",
        placeholder="/path/to/source_video.mp4",
    )
    selected_video = Path(custom_path).expanduser().resolve() if custom_path.strip() else None
    if selected_video:
        st.caption(str(selected_video))
if st.button(
    "Autofill from filename", help="Parse Show_SxxExx pattern from the selected video filename"
):
    video_for_autofill = selected_video or Path(
        st.session_state.get("process_video_custom", "") or ""
    )
    if video_for_autofill:
        guess = _guess_episode_from_path(video_for_autofill.name)
    else:
        guess = None
    if guess:
        st.session_state["process_show_slug"] = guess["show_slug"]
        st.session_state["process_episode_id"] = guess["episode_id"]
        st.session_state["process_flash"] = {
            "variant": "info",
            "message": f"Autofilled episode details from '{guess['source']}'.",
        }
        st.rerun()
    else:
        st.warning("Unable to parse show + episode from the selected filename.")
with st.expander("Audio Preset (Advanced)", expanded=False):
    st.caption(
        "Override preprocessing behaviour. These settings are recorded in metadata and may influence future pipeline runs."
    )
    preset_path = st.text_input("Preset file", key="process_preset_path")
    toggle_cols = st.columns(2)
    enable_separation = toggle_cols[0].checkbox(
        "Vocal separation",
        key="process_enable_separation",
        help="⚠️ Takes 3-5 minutes per episode. Optional for transcription."
    )
    enable_enhancement = toggle_cols[1].checkbox(
        "Enhancement",
        key="process_enable_enhancement",
        help="⚠️ Takes 2-4 minutes per episode. Optional for transcription."
    )
    if enable_separation or enable_enhancement:
        st.warning(
            "⏱️ Vocal separation and enhancement are CPU/GPU intensive. "
            "Processing may take 5-10 minutes with no visible progress updates. "
            "The process is working - please wait for completion.",
            icon="⚠️"
        )
    provider = st.selectbox(
        "Enhancer provider",
        options=["resemble", "clearervoice"],
        key="process_enhancer_provider",
        format_func=lambda value: (
            "Resemble (default)" if value == "resemble" else "ClearerVoice Studio"
        ),
        help="Select ClearerVoice after installing the `clearvoice` package to run the unified ClearerVoice Studio stack.",
    )
    if provider == "clearervoice" and not _CLEARERVOICE_AVAILABLE:
        st.warning(
            "ClearerVoice is not installed. Run `pip install clearvoice` to enable this provider.",
            icon="⚠️",
        )
    rate_cols = st.columns(2)
    sample_rate = rate_cols[0].number_input(
        "Sample rate (Hz)",
        min_value=8000,
        max_value=96000,
        value=st.session_state["process_sample_rate"],
        step=1000,
        key="process_sample_rate",
    )
    channels = rate_cols[1].selectbox(
        "Channels",
        options=[1, 2],
        format_func=lambda value: "Mono" if value == 1 else "Stereo",
        key="process_channels",
    )
    target_lufs = st.number_input(
        "Normalization target (LUFS)",
        value=float(st.session_state["process_target_lufs"]),
        step=0.5,
        key="process_target_lufs",
    )


def _validate_inputs(video: Path | None) -> list[str]:
    errors: list[str] = []
    if not show_slug.strip():
        errors.append("Show slug is required.")
    if not episode_id.strip():
        errors.append("Episode ID is required.")
    if video is None:
        errors.append("Select a source video file from the dropdown or provide a custom path.")
    elif not video.exists():
        errors.append(f"Source video file not found: {video}")
    preset_candidate = Path(preset_path).expanduser()
    if not preset_candidate.exists():
        errors.append(f"Preset file not found: {preset_candidate}")
    return errors


action_cols = st.columns((2, 1))
with action_cols[0]:
    run_create_audio = st.button("CREATE AUDIO", type="primary", use_container_width=True)
with action_cols[1]:
    cancel_create_audio = st.button(
        "Cancel CREATE AUDIO",
        type="secondary",
        use_container_width=True,
        help="Stop any currently running CREATE AUDIO commands.",
    )
if cancel_create_audio:
    cancelled = cancel_active_audio_runs()
    if cancelled:
        st.warning("Sent cancellation signal to the active CREATE AUDIO run.")
    else:
        st.info("No CREATE AUDIO run detected.")
if run_create_audio:
    issues = _validate_inputs(selected_video)
    if issues:
        for issue in issues:
            st.error(issue)
    else:
        video = selected_video.expanduser() if selected_video else None
        if video is None:
            st.error("Select a source video before running CREATE AUDIO.")
            st.stop()
        preset = Path(preset_path).expanduser()
        status_indicator = st.status("Queued CREATE AUDIO run…", expanded=True)
        progress_bar = st.progress(0.0)
        progress_entries: list[dict[str, Any]] = []
        episode_dir: Path | None = None

        def _append_log(message: str, fraction: float | None = None, level: str = "info") -> None:
            timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            bounded_fraction: float | None = None
            percent_value: int | None = None
            if fraction is not None:
                bounded_fraction = max(0.0, min(1.0, float(fraction)))
                percent_value = int(round(bounded_fraction * 100))
                progress_bar.progress(bounded_fraction)
                display_message = f"{percent_value}% · {message}"
            else:
                display_message = message
            status_indicator.write(display_message)
            progress_entries.append(
                {
                    "message": message,
                    "fraction": bounded_fraction,
                    "percent": percent_value,
                    "level": level,
                    "timestamp": timestamp,
                }
            )

        _append_log("CREATE AUDIO run queued.", 0.0)

        try:
            _append_log("Preparing episode scaffold…", 0.05)
            episode_dir = ensure_episode_scaffold(
                show_slug.strip(), episode_id.strip(), video, preset
            )
            _append_log("Episode scaffold ready.", 0.12)
            metadata_updates = {
                "show_config": show_config_path.strip() or None,
                "preprocessing": {
                    "enable_separation": enable_separation,
                    "enable_enhancement": enable_enhancement,
                    "provider": provider,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "target_lufs": target_lufs,
                },
            }
            update_metadata(episode_dir, metadata_updates)
            _append_log("Running audio extraction and preprocessing…", 0.2)

            def _report_progress(message: str, fraction: float | None = None) -> None:
                _append_log(message, fraction)

            try:
                artifacts = create_audio(episode_dir, video, preset, progress=_report_progress)
            except TypeError as exc:
                if "progress" in str(exc):
                    _append_log(
                        "Progress callback not supported by create_audio(); continuing without live updates.",
                        None,
                        level="warning",
                    )
                    artifacts = create_audio(episode_dir, video, preset)
                else:
                    raise
        except AudioPreprocessingCancelled as exc:
            detail = exc.stderr or str(exc) or "CREATE AUDIO run cancelled by user request."
            _append_log(detail, None, level="warning")
            status_indicator.update(label="CREATE AUDIO run cancelled.", state="error")
            st.info("CREATE AUDIO run cancelled.")
            partial_artifacts = {
                key: str(Path(value))
                for key, value in (getattr(exc, "artifacts", {}) or {}).items()
                if value
            }
            if partial_artifacts:
                st.caption("Artifacts generated before cancellation:")
                with st.expander("Partial artifacts", expanded=False):
                    st.json(partial_artifacts)
            st.session_state["process_last_log"] = {
                "episode_dir": str(episode_dir) if episode_dir else "",
                "error": detail,
                "artifacts": partial_artifacts,
                "messages": progress_entries,
                "status": "cancelled",
            }
            st.stop()
        except AudioPreprocessingError as exc:
            detail = exc.stderr or str(exc)
            _append_log(detail or "Audio preprocessing failed.", None, level="error")
            status_indicator.update(label="Audio preprocessing failed.", state="error")
            st.error(f"Audio preprocessing failed: {detail}")
            partial_artifacts = {
                key: str(Path(value))
                for key, value in (getattr(exc, "artifacts", {}) or {}).items()
                if value
            }
            if partial_artifacts:
                st.warning(
                    "Partial audio artifacts were produced and saved. "
                    "Fix the failure and rerun CREATE AUDIO to regenerate them."
                )
                with st.expander("Partial artifacts", expanded=False):
                    st.json(partial_artifacts)
            st.session_state["process_last_log"] = {
                "episode_dir": str(episode_dir) if episode_dir else "",
                "error": detail,
                "artifacts": partial_artifacts,
                "messages": progress_entries,
                "status": "error",
            }
        except Exception as exc:
            detail = str(exc)
            _append_log(detail, None, level="error")
            status_indicator.update(label="CREATE AUDIO failed.", state="error")
            st.error(f"Audio preprocessing failed: {detail}")
            st.session_state["process_last_log"] = {
                "episode_dir": str(episode_dir) if episode_dir else "",
                "error": detail,
                "artifacts": {},
                "messages": progress_entries,
                "status": "error",
            }
        else:
            label = f"{show_slug.strip()} · {episode_id.strip()}"
            st.session_state[SESSION_LABEL_KEY] = label
            st.session_state[SESSION_DIR_KEY] = str(episode_dir)
            relative_artifacts: dict[str, str] = {}
            for name, path in artifacts.items():
                if not path:
                    continue
                try:
                    rel_path = Path(path).resolve().relative_to(episode_dir)
                except ValueError:
                    rel_path = Path(path)
                relative_artifacts[name] = str(rel_path)
            if not any((entry.get("percent") == 100) for entry in progress_entries):
                _append_log("Audio preprocessing stage complete.", 1.0, level="success")
            _append_log("Audio artifacts saved.", None, level="success")
            status_indicator.update(
                label=f"Audio preprocessing completed for {label}.",
                state="complete",
            )
            progress_bar.progress(1.0)
            st.session_state["process_last_log"] = {
                "episode_dir": str(episode_dir),
                "artifacts": relative_artifacts,
                "messages": progress_entries,
                "status": "success",
            }
            st.session_state["process_flash"] = {
                "variant": "success",
                "message": f"Audio artifacts created for {label}.",
                "artifacts": relative_artifacts,
            }
            st.rerun()
st.subheader("Existing Audio")
if active_dir:
    audio_options = {
        "Enhanced (mix, with background)": resolve_artifact_path(active_dir, "audio_enhanced_mix"),
        "Enhanced Vocals (no background)": resolve_artifact_path(
            active_dir, "audio_enhanced_vocals"
        )
        or resolve_artifact_path(active_dir, "audio_enhanced"),
        "Vocals (no background)": resolve_artifact_path(active_dir, "audio_vocals"),
        "Raw Extracted": resolve_artifact_path(active_dir, "audio_extracted"),
    }
    available = [(label, path) for label, path in audio_options.items() if path and path.exists()]
    if available:
        labels = [label for label, _ in available]
        preferred = "Enhanced Vocals (no background)"
        if preferred not in labels:
            preferred = labels[0]
        default_label = st.session_state.get("process_audio_choice") or preferred
        if default_label not in labels:
            default_label = labels[0]
        choice = st.selectbox(
            "Available audio files",
            labels,
            index=labels.index(default_label),
            key="process_audio_choice",
        )
        selected_path = dict(available).get(choice)
        if selected_path:
            render_artifact_badge(selected_path, selected_path.name)
            st.session_state["process_selected_audio_path"] = str(selected_path)
        else:
            st.warning("Selected audio file is no longer available.")
    else:
        st.caption("No audio artifacts detected yet for the active episode.")
else:
    st.caption("Select or create an episode to inspect audio artifacts.")
if active_dir:
    with st.expander("Episode Metadata", expanded=False):
        st.json(active_metadata or read_metadata(active_dir) or {})


def _trigger_rerun() -> None:
    """Compat wrapper for Streamlit rerun API."""
    try:
        st.rerun()  # type: ignore[attr-defined]
    except AttributeError:
        st.rerun()
