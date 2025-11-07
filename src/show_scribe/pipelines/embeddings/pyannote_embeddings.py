"""Speaker embedding extraction using Pyannote models."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ...utils.audio_io import AudioClip
from ...utils.hf_hub_compat import ensure_use_auth_token_compat
from ...utils.logging import get_logger

LOGGER = get_logger(__name__)
ensure_use_auth_token_compat()


def _ensure_hf_token_env(token: str) -> None:
    """Populate common Hugging Face auth environment variables when absent."""
    aliases = (
        "HUGGINGFACEHUB_API_TOKEN",
        "HF_TOKEN",
        "HF_API_TOKEN",
        "HUGGINGFACE_API_KEY",
        "PYANNOTE_TOKEN",
        "PYANNOTE_AUTH_TOKEN",
    )
    for env_var in aliases:
        if not os.environ.get(env_var):
            os.environ[env_var] = token


try:  # pragma: no cover - optional dependency
    from pyannote.audio import Inference as _PyannoteInference
    from pyannote.audio import Model as _PyannoteModel
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    _PyannoteInference = None
    _PyannoteModel = None
    _IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - optional dependency
    _IMPORT_ERROR = None

__all__ = [
    "PyannoteEmbeddingBackend",
    "PyannoteEmbeddingError",
    "PyannoteEmbeddingSettings",
    "build_pyannote_embedding_backend",
]


class PyannoteEmbeddingError(RuntimeError):
    """Raised when Pyannote embedding extraction cannot be performed."""


@dataclass(slots=True)
class PyannoteEmbeddingSettings:
    """Configuration used when constructing the Pyannote embedding backend."""

    model: str
    auth_token: str
    target_sample_rate: int = 16_000


class PyannoteEmbeddingBackend:
    """Thin wrapper around ``pyannote.audio`` embedding extraction."""

    def __init__(self, settings: PyannoteEmbeddingSettings) -> None:
        if _PyannoteModel is None or _PyannoteInference is None:
            assert _IMPORT_ERROR is not None
            raise PyannoteEmbeddingError(
                "pyannote.audio is not installed. Install the 'pyannote.audio' extra "
                "to enable speaker embedding extraction."
            ) from _IMPORT_ERROR

        self.settings = settings
        self.target_sample_rate = int(settings.target_sample_rate)
        self._lock = threading.Lock()
        self._inference: _PyannoteInference | None = None

    def encode(self, clip: AudioClip) -> np.ndarray:
        """Return a single embedding vector for ``clip``."""
        inference = self._ensure_inference()

        waveform = clip.samples
        # Ensure mono 1D first, then convert to (channels, time) torch tensor as required by pyannote
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0, dtype=np.float32)
        waveform = waveform.astype(np.float32, copy=False)

        if clip.sample_rate != self.target_sample_rate:
            raise PyannoteEmbeddingError(
                f"Expected sample rate {self.target_sample_rate}Hz, "
                f"received {clip.sample_rate}Hz."
            )

        # Convert to torch tensor with shape (channel, time)
        try:
            import torch  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - optional dependency
            raise PyannoteEmbeddingError(
                "Pyannote embeddings require PyTorch. Install 'torch' to proceed."
            ) from exc

        tensor = torch.from_numpy(waveform)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 2:
            raise PyannoteEmbeddingError(
                f"Unexpected waveform shape {tuple(tensor.shape)}; expected 1D or 2D (C, T)."
            )

        result = inference({"waveform": tensor, "sample_rate": clip.sample_rate})
        vector = self._to_numpy(result)
        if vector.ndim > 1:
            vector = vector.mean(axis=0)
        return vector.astype(np.float32, copy=False)

    def _ensure_inference(self) -> _PyannoteInference:
        with self._lock:
            if self._inference is not None:
                return self._inference

            LOGGER.debug("Loading Pyannote embedding model '%s'.", self.settings.model)
            if self.settings.auth_token:
                _ensure_hf_token_env(self.settings.auth_token)
            try:
                # Pass token explicitly and also ensure env vars are set for robustness
                model = _PyannoteModel.from_pretrained(
                    self.settings.model,
                    use_auth_token=self.settings.auth_token,
                )
            except Exception as exc:  # pragma: no cover - dependency specific
                raise PyannoteEmbeddingError(
                    f"Failed to load Pyannote embedding model '{self.settings.model}': {exc}"
                ) from exc

            # Defensive: some pyannote versions may return None instead of raising
            if model is None:  # pragma: no cover - defensive against dependency quirks
                raise PyannoteEmbeddingError(
                    "Failed to load Pyannote embedding model: received None. "
                    "Ensure the model identifier is correct, your Hugging Face token has access "
                    f"to '{self.settings.model}', and you have accepted any gated model terms."
                )

            self._inference = _PyannoteInference(model, window="whole")
            return self._inference

    @staticmethod
    def _to_numpy(result: Any) -> np.ndarray:
        """Convert various Pyannote outputs into a NumPy array."""
        if hasattr(result, "data"):
            array = np.asarray(result.data)
        elif hasattr(result, "numpy"):
            array = np.asarray(result.numpy())
        else:
            array = np.asarray(result)
        if array.size == 0:
            raise PyannoteEmbeddingError("Received empty embedding from Pyannote inference.")
        return array


def build_pyannote_embedding_backend(config: Mapping[str, object]) -> PyannoteEmbeddingBackend:
    """Construct a Pyannote embedding backend from the project configuration."""
    providers = config.get("providers") if isinstance(config, Mapping) else {}
    if not isinstance(providers, Mapping):
        raise PyannoteEmbeddingError("Configuration is missing the 'providers' section.")

    pyannote_cfg = providers.get("pyannote")
    if not isinstance(pyannote_cfg, Mapping):
        raise PyannoteEmbeddingError("Configuration is missing 'providers.pyannote'.")

    model_name = str(pyannote_cfg.get("embedding_model", "pyannote/embedding"))
    auth_env = str(pyannote_cfg.get("auth_token_env", "PYANNOTE_TOKEN"))
    candidate_vars = [
        auth_env,
        "PYANNOTE_AUTH_TOKEN",
        "PYANNOTE_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "HUGGINGFACE_API_KEY",
        "HF_TOKEN",
        "HF_API_TOKEN",
    ]
    token = None
    for env_var in candidate_vars:
        token = os.environ.get(env_var)
        if token:
            break
    if not token:
        unique_vars = ", ".join(dict.fromkeys(candidate_vars))
        raise PyannoteEmbeddingError(
            "Pyannote authentication token not available. "
            f"Set one of the environment variables: {unique_vars}."
        )

    audio_cfg = config.get("audio") if isinstance(config, Mapping) else {}
    if isinstance(audio_cfg, Mapping):
        target_rate = int(audio_cfg.get("sample_rate", 16_000))
    else:
        target_rate = 16_000

    settings = PyannoteEmbeddingSettings(
        model=model_name,
        auth_token=token,
        target_sample_rate=target_rate,
    )
    return PyannoteEmbeddingBackend(settings)
