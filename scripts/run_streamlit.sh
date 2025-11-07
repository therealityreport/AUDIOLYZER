#!/usr/bin/env bash
# Run (or restart) the Show‑Scribe Streamlit UI with sensible defaults.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR%/scripts}"
APP_PATH="$REPO_ROOT/ui/streamlit_app/app.py"
VENV_BIN="$REPO_ROOT/.venv/bin"

# Defaults (override via flags or env)
PORT="${PORT:-8501}"
ENVIRONMENT="${ENV:-dev}"
OPEN_BROWSER=1
DO_KILL=1

usage() {
  cat <<EOF
Usage: $(basename "$0") [--port N] [--env NAME] [--no-browser] [--browser] [--no-kill] [--kill]

Options:
  --port N         Streamlit port (default: $PORT)
  --env NAME       Config environment (default: $ENVIRONMENT)
  --no-browser     Do not auto-open a browser
  --browser        Auto-open a browser (default)
  --no-kill        Do not kill any existing server on the port
  --kill           Kill any existing server on the port (default)
  -h, --help       Show this help and exit

Environment overrides: PORT, ENV
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"; shift 2 ;;
    --env)
      ENVIRONMENT="$2"; shift 2 ;;
    --no-browser)
      OPEN_BROWSER=0; shift ;;
    --browser)
      OPEN_BROWSER=1; shift ;;
    --no-kill)
      DO_KILL=0; shift ;;
    --kill)
      DO_KILL=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

# Prefer project venv if available
if [[ -x "$VENV_BIN/python" ]]; then
  PYTHON="$VENV_BIN/python"
else
  PYTHON="python3"
fi

if [[ ! -f "$APP_PATH" ]]; then
  echo "Streamlit app not found: $APP_PATH" >&2
  exit 2
fi

# Optionally kill any existing server on the chosen port
if [[ "$DO_KILL" -eq 1 ]]; then
  if command -v lsof >/dev/null 2>&1; then
    pids=$(lsof -ti tcp:"$PORT" || true)
    if [[ -n "$pids" ]]; then
      echo "Stopping existing server(s) on port $PORT: $pids"
      kill -15 $pids || true
      sleep 0.5
      # force kill if still alive
      pids=$(lsof -ti tcp:"$PORT" || true)
      [[ -n "$pids" ]] && kill -9 $pids || true
    fi
  fi
fi

echo "Starting Streamlit UI (env=$ENVIRONMENT, port=$PORT)"

# Use the CLI launcher which sets PYTHONPATH and config automatically
if [[ "$OPEN_BROWSER" -eq 1 ]]; then
  exec "$PYTHON" -m show_scribe.cli ui --env "$ENVIRONMENT" --port "$PORT" --browser
else
  exec "$PYTHON" -m show_scribe.cli ui --env "$ENVIRONMENT" --port "$PORT" --no-browser
fi
