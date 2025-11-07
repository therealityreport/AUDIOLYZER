"""Compatibility helpers for newer huggingface_hub releases."""

from __future__ import annotations

import inspect
import os
from functools import wraps
from typing import Any, Callable, TypeVar, overload

_PATCHED = False
_TOKEN_ENV_VARS = (
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGINGFACE_API_KEY",
    "HF_TOKEN",
    "HF_API_TOKEN",
    "PYANNOTE_TOKEN",
    "PYANNOTE_AUTH_TOKEN",
)

F = TypeVar("F", bound=Callable[..., Any])


def ensure_use_auth_token_compat() -> None:
    """Ensure huggingface_hub accepts the deprecated ``use_auth_token`` kwarg."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        import huggingface_hub  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        return

    _patch_callable(huggingface_hub, "hf_hub_download")
    _patch_callable(huggingface_hub, "snapshot_download")
    _PATCHED = True


@overload
def _patch_callable(module: Any, attribute: str) -> None: ...


def _patch_callable(module: Any, attribute: str) -> None:
    original: Any = getattr(module, attribute, None)
    if not callable(original):
        return

    try:
        parameters = inspect.signature(original).parameters
    except (TypeError, ValueError):  # pragma: no cover - defensive
        parameters = {}

    if "use_auth_token" in parameters:
        return

    @wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        token = _pop_use_auth_token(kwargs)
        if token is not None and "token" not in kwargs:
            kwargs["token"] = token
        return original(*args, **kwargs)

    setattr(module, attribute, wrapper)


def _pop_use_auth_token(kwargs: dict[str, Any]) -> str | None:
    if "use_auth_token" not in kwargs:
        return None
    token = kwargs.pop("use_auth_token")
    if token in (None, False):
        return None
    if token is True:
        return _resolve_env_token()
    return str(token)


def _resolve_env_token() -> str | None:
    for env_var in _TOKEN_ENV_VARS:
        value = os.environ.get(env_var)
        if value:
            return value
    return None
