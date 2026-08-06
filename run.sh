#!/usr/bin/env bash
# EduPilot supervisor wrapper
# Restarts streamlit automatically on any exit, logging timestamps and exit codes.
# Never use waitForPort — the workflow is configured without it (known broken for port 8000).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESTART_DELAY=2
attempt=0

while true; do
    attempt=$((attempt + 1))
    echo "[supervisor] $(date -u +%Y-%m-%dT%H:%M:%SZ) attempt=$attempt starting streamlit..."
    python -m streamlit run app.py \
        --server.port 8000 \
        --server.headless true \
        --server.enableCORS false \
        --server.enableXsrfProtection false \
        --server.baseUrlPath /edupilot \
        || true
    exit_code=$?
    echo "[supervisor] $(date -u +%Y-%m-%dT%H:%M:%SZ) streamlit exited with code=$exit_code attempt=$attempt — restarting in ${RESTART_DELAY}s"
    sleep "$RESTART_DELAY"
done
