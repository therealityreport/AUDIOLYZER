from typing import Any

class SoundFile:
    samplerate: int
    channels: int
    format: str | None
    subtype: str | None
    frames: int

    def __init__(self, file: str, mode: str = "r") -> None: ...
    def __enter__(self) -> SoundFile: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None: ...

def read(
    file: str,
    *,
    dtype: str = ...,
    always_2d: bool = ...,
) -> tuple[Any, int]: ...
def write(
    file: str,
    data: Any,
    samplerate: int,
    *,
    subtype: str = ...,
) -> None: ...

__all__ = ["SoundFile", "read", "write"]
