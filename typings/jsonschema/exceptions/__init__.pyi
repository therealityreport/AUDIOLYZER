from collections.abc import Iterable
from typing import Any

class ValidationError(Exception):
    path: Iterable[Any]
    message: str

    def __init__(self, message: str = "", path: Iterable[Any] | None = ...) -> None: ...

__all__ = ["ValidationError"]
