from collections.abc import Iterable, Mapping
from typing import Any

from jsonschema.exceptions import ValidationError

class Draft7Validator:
    schema: Mapping[str, Any]

    def __init__(self, schema: Mapping[str, Any]) -> None: ...
    def iter_errors(self, instance: Mapping[str, Any]) -> Iterable[ValidationError]: ...

__all__ = ["Draft7Validator"]
