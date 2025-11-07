from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_pipeline as run_pipeline
from show_scribe.pipelines.orchestrator import JobStatus


class StubPaths:
    def __init__(self, base: Path) -> None:
        self.data_root = base

    def show_root(self, show_name: str) -> Path:
        return self.data_root / "shows" / show_name

    def episode_directory(self, show_name: str, episode_id: str) -> Path:
        return self.show_root(show_name) / "episodes" / episode_id


class StubContext:
    def __init__(self, base: Path) -> None:
        self.config: dict[str, object] = {"audio_preprocessing": {"enable": False}}
        self.paths = StubPaths(base)
        self.environment = "test"


@pytest.fixture
def media_fixture() -> Path:
    return Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "video" / "test_clip.mp4"


def test_cli_uses_enhanced_audio(
    tmp_path: Path, media_fixture: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode_dir = tmp_path / "episode"
    show_config = (
        Path(__file__).resolve().parents[3] / "data" / "shows" / "RHOBH" / "show_config.json"
    )

    context = StubContext(tmp_path)
    monkeypatch.setattr(run_pipeline, "_load_context", lambda env, overrides=None: context)
    monkeypatch.setattr(run_pipeline, "ensure_ffmpeg_available", lambda: None)

    def _fake_extract_and_enhance(input_path: Path, target_dir: Path, preset: Path):
        extracted = target_dir / "audio_extracted.wav"
        vocals = target_dir / "audio_vocals.wav"
        enhanced = target_dir / "audio_enhanced_vocals.wav"
        enhanced_mix = target_dir / "audio_enhanced_mix.wav"
        report = target_dir / "preprocessing_report.json"
        target_dir.mkdir(parents=True, exist_ok=True)
        extracted.write_bytes(b"extracted")
        vocals.write_bytes(b"vocals")
        enhanced.write_bytes(b"enhanced")
        enhanced_mix.write_bytes(b"mix")
        report.write_text("{}", encoding="utf-8")
        return {
            "extracted": extracted,
            "vocals": vocals,
            "enhanced_vocals": enhanced,
            "enhanced": enhanced,  # legacy key for compatibility
            "enhanced_mix": enhanced_mix,
            "report": report,
        }

    monkeypatch.setattr(run_pipeline, "extract_and_enhance", _fake_extract_and_enhance)

    captured: dict[str, object] = {}

    class FakeOrchestrator:
        def __init__(
            self, *, context, max_workers: int, max_queue: int
        ) -> None:  # noqa: D401, ANN001 - test stub
            captured["context"] = context
            captured["orchestrator"] = self
            self.submitted_input: Path | None = None
            self._job = SimpleNamespace(
                job_id="job-1",
                episode_id="RHOBH_S13E01",
                current_stage=None,
                progress=1.0,
                message="",
                status=JobStatus.COMPLETED,
            )

        def submit(
            self,
            input_path: Path,
            *,
            episode_id: str,
            show_config_path: Path | None,
            metadata: dict[str, object] | None,
            progress_handler=None,
        ):  # noqa: ANN001 - test stub signature
            self.submitted_input = Path(input_path)
            captured["metadata"] = metadata
            return self._job

        def wait(self, job_id: str) -> None:  # noqa: D401
            return None

        def shutdown(self) -> None:  # noqa: D401
            return None

        def get_job(self, job_id: str):  # noqa: ANN001 - test stub signature
            return self._job

    monkeypatch.setattr(run_pipeline, "PipelineOrchestrator", FakeOrchestrator)

    exit_code = run_pipeline.main(
        [
            "--input",
            str(media_fixture),
            "--episode-id",
            "RHOBH_S13E01",
            "--show-config",
            str(show_config),
            "--preprocess",
            "--preset",
            "reality_tv",
            "--episode-dir",
            str(episode_dir),
        ]
    )

    assert exit_code == 0
    orchestrator_instance = captured.get("orchestrator")
    assert isinstance(orchestrator_instance, FakeOrchestrator)
    assert orchestrator_instance.submitted_input == (episode_dir / "audio_enhanced_vocals.wav")
    metadata = captured.get("metadata")
    assert isinstance(metadata, dict)
    assert metadata.get("episode_dir_override") == str(episode_dir.resolve())
