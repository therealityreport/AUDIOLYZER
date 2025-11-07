from collections.abc import Iterable
from typing import Any

class WhisperModel:
    def __init__(
        self,
        model_path: str,
        *,
        device: str = ...,
        compute_type: str = ...,
        download_root: str | None = ...,
    ) -> None: ...
    def transcribe(
        self,
        audio_path: str,
        **decode_options: Any,
    ) -> tuple[Iterable[Any], Any]: ...

__all__ = ["WhisperModel"]
