from collections.abc import Mapping
from typing import Any

class Pipeline:
    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        *,
        revision: str | None = ...,
        hparams_file: str | None = ...,
        token: str | None = ...,
        cache_dir: str | None = ...,
    ) -> Pipeline | None: ...
    def parameters(self) -> Mapping[str, Any]: ...
    def instantiate(self, params: Mapping[str, Any]) -> None: ...
    def to(self, device: Any) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

__all__ = ["Pipeline"]
