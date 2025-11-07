from collections.abc import Iterable
from typing import Any

def resample_poly(
    data: Iterable[Any],
    up: int,
    down: int,
) -> Any: ...
def stft(
    x: Iterable[Any],
    fs: int,
    *,
    nperseg: int = ...,
    noverlap: int | None = ...,
    boundary: str | None = ...,
) -> tuple[Any, Any, Any]: ...

__all__ = ["resample_poly", "stft"]
