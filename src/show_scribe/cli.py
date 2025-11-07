"""Command-line entrypoints for Show-Scribe."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from show_scribe.pipelines.orchestrator import (
    JobStatus,
    PipelineJob,
    PipelineOrchestrator,
    build_default_context,
)

app = typer.Typer(help="Process and analyze podcast episodes.")


def _progress_printer(job: PipelineJob) -> None:
    stage = job.current_stage or "initialising"
    percent = f"{job.progress * 100:5.1f}%"
    message = job.message or ""
    typer.echo(f"[{stage}] {percent} {message}")


@app.command()
def process(
    input_path: str = typer.Argument(..., help="Source media file (video or audio)."),
    output_dir: Optional[Path] = typer.Argument(
        None,
        metavar="OUTPUT_DIR",
        help="Optional directory override for episode artifacts.",
    ),
    episode_id: Optional[str] = typer.Option(
        None,
        "--episode-id",
        "-e",
        help="Episode identifier (defaults to input file stem).",
    ),
    show_config: Optional[Path] = typer.Option(
        None,
        "--show-config",
        "-c",
        help="Path to show_config.json when it cannot be inferred.",
    ),
    env: str = typer.Option(
        "dev",
        "--env",
        help="Configuration environment to load (default: dev).",
    ),
    max_workers: int = typer.Option(1, "--max-workers", help="Maximum concurrent workers."),
    max_queue: int = typer.Option(4, "--max-queue", help="Queue depth before rejecting jobs."),
    watch: bool = typer.Option(
        True,
        "--watch/--no-watch",
        help="Stream stage progress to the console.",
    ),
) -> None:
    """Kick off the end-to-end processing pipeline for a single episode."""

    input_path_obj = Path(input_path).expanduser().resolve()
    if not input_path_obj.exists():
        typer.echo(f"Input file not found: {input_path_obj}", err=True)
        raise typer.Exit(code=2)

    episode_slug = episode_id or input_path_obj.stem

    try:
        context = build_default_context(env)
    except Exception as exc:  # pragma: no cover - configuration safety
        typer.echo(f"Failed to load configuration: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    orchestrator = PipelineOrchestrator(
        context=context,
        max_workers=max_workers,
        max_queue=max_queue,
    )

    metadata: dict[str, str] = {}
    if output_dir is not None:
        metadata["episode_dir_override"] = str(output_dir.expanduser().resolve())

    progress_handler = _progress_printer if watch else None
    try:
        job = orchestrator.submit(
            input_path_obj,
            episode_id=episode_slug,
            show_config_path=show_config,
            metadata=metadata,
            progress_handler=progress_handler,
        )
        orchestrator.wait(job.job_id)
    except KeyboardInterrupt:  # pragma: no cover - interactive guard
        typer.echo("Interrupted, shutting down workers…", err=True)
        raise typer.Exit(code=130) from None
    finally:
        orchestrator.shutdown()

    if job.status != JobStatus.COMPLETED:
        error_message = job.error or job.message or "Pipeline failed."
        typer.echo(error_message, err=True)
        raise typer.Exit(code=1)

    typer.echo(job.message or "Pipeline completed successfully.")


@app.command()
def download_models(force: bool = False) -> None:
    """Download or refresh required ML models."""
    raise NotImplementedError("Model download pipeline pending implementation.")


@app.command()
def ui(
    env: str = typer.Option(
        "dev",
        "--env",
        help="Configuration environment to load (default: dev).",
    ),
    port: Optional[int] = typer.Option(
        None,
        "--port",
        help="Override Streamlit server port (defaults to config value).",
    ),
    open_browser: bool = typer.Option(
        True,
        "--browser/--no-browser",
        help="Whether to launch the default browser automatically.",
    ),
) -> None:
    """Launch the Streamlit review console."""

    try:
        import streamlit.web.bootstrap as streamlit_bootstrap
    except ImportError as exc:  # pragma: no cover - optional dependency
        typer.echo(
            "Streamlit is not installed. Install the 'streamlit' extra to enable the UI.",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    context = build_default_context(env)
    config = context.config

    default_port = 8501
    auto_browser = True
    ui_section = config.get("ui")
    if isinstance(ui_section, dict):
        streamlit_cfg = ui_section.get("streamlit")
        if isinstance(streamlit_cfg, dict):
            default_port = int(streamlit_cfg.get("port", default_port))
            auto_browser = bool(streamlit_cfg.get("auto_open_browser", auto_browser))
    if port is None:
        port = default_port
    if not open_browser:
        auto_browser = False

    app_path = Path(__file__).resolve().parents[2] / "ui" / "streamlit_app" / "app.py"
    if not app_path.exists():
        typer.echo(f"Streamlit application not found at {app_path}", err=True)
        raise typer.Exit(code=2)

    import os

    flag_options = {
        "server.port": port,
        "server.headless": not auto_browser,
    }

    pythonpath = os.environ.get("PYTHONPATH", "")
    project_root = context.paths.project_root
    path_entries = [str(project_root / "src"), str(project_root)] + (
        [pythonpath] if pythonpath else []
    )
    os.environ["PYTHONPATH"] = os.pathsep.join(path_entries)

    typer.echo(f"Starting Streamlit UI from {app_path} on port {port} (env={env})")
    streamlit_bootstrap.run(
        str(app_path),
        is_hello=False,
        args=[],
        flag_options=flag_options,
    )


def main() -> None:
    """Entrypoint used by the console script."""
    app()


if __name__ == "__main__":
    main()
