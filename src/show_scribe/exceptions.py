"""Exception types for Show-Scribe."""

from __future__ import annotations

__all__ = ["AudioPreprocessingCancelled", "AudioPreprocessorCancelled"]


class AudioPreprocessingCancelled(Exception):
    """
    Raised by UI-layer code to signal that the user intentionally cancelled
    a long-running audio preprocessing task (e.g., clicked 'Cancel').
    Catch this in Streamlit pages to stop the run cleanly without a traceback.
    """

    pass


# Back-compat alias for earlier misspelling
AudioPreprocessorCancelled = AudioPreprocessingCancelled
