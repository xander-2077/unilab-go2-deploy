#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ "$#" -eq 0 ]; then
  set -- python deploy.py
fi

# exec /home/unitree/.local/bin/uv run --env-file .env "$@"
exec /home/unitree/.local/bin/uv run "$@"
