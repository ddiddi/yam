#!/bin/zsh
# Dance the YAM arm.
#
#   ./run_dance.sh                 slow (speed 0.5), plays the choreography once
#   ./run_dance.sh 1.0             normal speed
#   ./run_dance.sh 0.3 60          speed 0.3, dance for 60 s (loops the choreography)
#   ./run_dance.sh 0.5 0 --sim     any extra flags go straight to dance.py (--amplitude 0.7, --gripper no_gripper ...)
#
# Ctrl-C at any time: the arm glides back to rest and torques off.
SPEED=${1:-0.5}
DURATION=${2:-0}
shift 2 2>/dev/null || shift $# 2>/dev/null
cd "$(dirname "$0")"
source .venv/bin/activate
exec python dance.py --speed "$SPEED" --duration "$DURATION" "$@"
