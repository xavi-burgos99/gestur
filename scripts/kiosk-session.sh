#!/usr/bin/env bash
set -u
cd /opt/gestur || exit 1
export GESTUR_CONFIG=/var/lib/gestur/config.json
export GESTUR_MODELS_DIR=/var/lib/gestur/models
export GESTUR_STATUS_PATH=/var/lib/gestur/runtime-status.json
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# xinit provides DISPLAY; use the accelerated Mesa driver chosen by KMS.
xset s off
xset -dpms
xset s noblank
openbox-session &
GESTUR_WM_PID=$!
trap 'kill "$GESTUR_WM_PID" 2>/dev/null || true' EXIT
while true; do
    /opt/gestur/.venv/bin/python -m gestur >> /var/log/gestur/viewer.log 2>&1
    GESTUR_RESULT=$?
    if [[ $GESTUR_RESULT == 0 ]]; then
        break
    fi
    # Allow a disconnected camera to recover without a busy restart loop.
    sleep 3
done
