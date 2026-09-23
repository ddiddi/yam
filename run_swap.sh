#!/bin/zsh
# ./run_swap.sh [N] [dataset]: N pick-and-place runs with a person swapping the object between runs.
# Run 1 picks whatever is in the zone; before every later run wait_new_object.py waits (arm parked) for the
# zone to change, go still for 4 s and hold an object, then counts down. Full pipeline each run: depth scene
# analysis, grasp across the short side, visual check, fused-view gate, tactile log, LeRobot recording.
# Stops on an arm/driver error, 2 runs in a row that did not complete, captures/live/STOP, or a 15 min wait.
cd "$(dirname "$0")"
N=${1:-10}; DS=${2:-datasets/yam_zone_swap}
mkdir -p captures/live
rm -f captures/live/STOP
fails=0
for k in $(seq 1 $N); do
  if [ -f captures/live/STOP ]; then echo "stopping: captures/live/STOP"; break; fi
  if [ $k -gt 1 ] || [ -n "$WAIT_FIRST" ]; then  # WAIT_FIRST=1: wait for a swap before run 1 too
    # the waiter's own exit status decides (piping it through a filter once made a killed waiter read as "go")
    .venv/bin/python -u wait_new_object.py --batch "$k/$N" --timeout 900 >> captures/live/swap_wait.log 2>&1
    if [ $? -ne 0 ]; then echo "stopping: no new object (waiter exited)"; break; fi
    grep -E "found:" captures/live/swap_wait.log | tail -1
    if [ -f captures/live/STOP ]; then echo "stopping: captures/live/STOP"; break; fi
  fi
  echo "=== run $k/$N ==="
  T=""; ls /dev/cu.usbmodem* >/dev/null 2>&1 && T="--tactile"
  YAM_BATCH="swap $k/$N" .venv/bin/python -u pick_place.py --run --object 0 --grasp side --shuttle --record "$DS" $T \
      > captures/live/swap_$k.log 2>&1
  st=$(python3 -c "import json;print(json.load(open('captures/live/state.json'))['status'])" 2>/dev/null)
  grep -E "depth says|target |^!!|gripper at|r\(skin, \|effort\|\) while|saved LeRobot" captures/live/swap_$k.log | grep -v RuntimeWarning | tail -6
  echo "run $k: $st"
  if [ "$st" = "done" ]; then fails=0; else fails=$((fails + 1)); fi
  if grep -qE "control loop died|fail to communicate|Motor error" captures/live/swap_$k.log; then echo "stopping: arm/driver error"; break; fi
  if [ $fails -ge 2 ]; then echo "stopping: 2 runs in a row did not complete"; break; fi
done
echo "=== swap batch finished ==="
