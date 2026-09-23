#!/bin/zsh
# ./run_batch.sh [N] [dataset]: N side-grasp pick-and-place runs back to back, all recorded into one LeRobot
# dataset; the object shuttles between two spots 5 cm apart. Each run's state goes to captures/live/state.json
# (the workspace map). Stops after 2 runs in a row that did not complete.
cd "$(dirname "$0")"
N=${1:-10}; DS=${2:-datasets/yam_side_pick_batch}
mkdir -p captures/live
fails=0
rm -f captures/live/STOP
for k in $(seq 1 $N); do
  if [ -f captures/live/STOP ]; then echo "stopping: captures/live/STOP"; break; fi  # touch it to stop between runs
  echo "=== run $k/$N ==="
  YAM_BATCH="$k/$N" .venv/bin/python -u pick_place.py --run --object 0 --grasp side --shuttle --record "$DS" \
      > captures/live/run_$k.log 2>&1
  st=$(python3 -c "import json;print(json.load(open('captures/live/state.json'))['status'])" 2>/dev/null)
  grep -E "^-> |^!!|holding the object|CLOSED ON NOTHING|saved LeRobot|^done" captures/live/run_$k.log | tail -4
  echo "run $k: $st"
  if [ "$st" = "done" ]; then fails=0; else fails=$((fails + 1)); fi
  if grep -qE "control loop died|fail to communicate|Motor error" captures/live/run_$k.log; then echo "stopping: arm/driver error"; break; fi
  if [ $fails -ge 2 ]; then echo "stopping: 2 runs in a row did not complete"; break; fi
  sleep 15  # time to touch captures/live/STOP between runs
done
echo "=== batch finished ==="
