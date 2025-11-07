from collections.abc import Iterator
from typing import Any

class Annotation:
    def itertracks(
        self,
        yield_label: bool = ...,
    ) -> Iterator[tuple[Any, int, Any]]: ...
    def __getitem__(self, key: tuple[Any, int]) -> Any: ...
    def get_timeline(self) -> Any: ...

__all__ = ["Annotation"]
