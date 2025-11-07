#!/usr/bin/env bash
#
# Setup script for installing FFmpeg on macOS machines used with Show-Scribe.

set -euo pipefail

if [[ "${OSTYPE:-}" != darwin* ]]; then
  echo "This installer only supports macOS. Detected OSTYPE='${OSTYPE:-unknown}'." >&2
  exit 1
fi

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

if ! command -v brew >/dev/null 2>&1; then
  cat <<'EOF' >&2
Homebrew is required to install FFmpeg automatically.
Install it with:
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
Then re-run this script.
EOF
  exit 1
fi

if brew ls --versions ffmpeg >/dev/null 2>&1; then
  if [[ "${FORCE}" -eq 1 ]]; then
    echo "Updating existing FFmpeg installation via Homebrew..."
    brew upgrade ffmpeg
  else
    echo "FFmpeg is already installed at $(command -v ffmpeg). Use --force to upgrade."
  fi
else
  echo "Installing FFmpeg via Homebrew..."
  brew install ffmpeg
fi

echo "Verifying FFmpeg binary..."
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "FFmpeg installation failed. Please check the Homebrew output above." >&2
  exit 1
fi

ffmpeg -version | head -n 1
echo "FFmpeg setup complete."
