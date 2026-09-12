#!/bin/sh
# Run podaddeduct with the project venv (system python has no deps).
exec "$(dirname "$0")/.venv/bin/python" -m podaddeduct "$@"
