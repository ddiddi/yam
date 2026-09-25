#!/bin/zsh
# ./scripts/dish_station_streak.sh [N] [x,y]: the Petri dish weigh-station cycle N times in a row (default 5).
# Before each run the dish is re-found on the tape (scripts/dish_find_rim.py, near where it was last); a run counts
# only if both grips passed their check, the run finished, and the dish is back within 2 cm of where it was picked.
# Stops at the first failure, an arm/driver error, or captures/live/STOP. Weigh photos -> captures/weigh_run<k>_*.jpg
cd "$(dirname "$0")/.."
export YAM_CAM_MAP="1:2,2:1"
N=${1:-5}; xy=${2:-0.3026,0.1057}
rm -f captures/live/STOP
ok=0; k=0; retries=0
while [ $ok -lt $N ]; do
  k=$((k + 1))
  [ -f captures/live/STOP ] && { echo "stopping: STOP file"; break; }
  new=$(.venv/bin/python scripts/dish_find_rim.py "$xy" --show captures/live/dish_before_$k.jpg 2>captures/live/dish_find_$k.err)
  [ -z "$new" ] && { echo "run $k: dish not found near ($xy): $(cat captures/live/dish_find_$k.err)"; break; }
  xy=$new
  echo "=== run $k (streak $ok/$N): dish at ($xy)"
  L=captures/live/station_dish_streak_$k.log
  YAM_BATCH="weigh station $((ok + 1))/$N" .venv/bin/python -u pick_place.py --run --station --weigh 8 --object 0 --grasp side \
    --side-only --side-cross-max 1.0 --side-tilt 42 40 --z-grasp 0.013 --max-width 0.092 --soft --grip-load 0.18 \
    --given-only --given "$xy,0.09,0.015" --record datasets/yam_weigh_station > $L 2>&1
  st=$(python3 -c "import json;print(json.load(open('captures/live/state.json'))['status'])")
  for f in captures/weigh_cam2_*.jpg(N); do mv $f captures/weigh_run${k}_${f:t:r:s/weigh_//}.jpg; done
  grips=$(grep -c "OK, lifting" $L)
  grep -E "soft contact|hold test|scale display|still holding|!!" $L | grep -v "moved\|tactile" | tail -6
  after=$(.venv/bin/python scripts/dish_find_rim.py "$xy" --show captures/live/dish_after_$k.jpg 2>/dev/null)
  d=$([ -n "$after" ] && python3 -c "import math;a=[float(v) for v in '$xy'.split(',')];b=[float(v) for v in '$after'.split(',')];print(f'{math.dist(a,b)*100:.1f}')" || echo "-")
  echo "run $k: status $st, grips $grips/2, dish back at ($after), $d cm from the pick point"
  if [ "$st" = "done" ] && [ "$grips" = "2" ] && [ "$d" != "-" ] && python3 -c "import sys;sys.exit(0 if $d <= 2.0 else 1)"; then
    ok=$((ok + 1)); echo "run $k: SUCCESS ($ok in a row)"; xy=$after
  elif [ "$grips" = "0" ] && [ "$d" != "-" ] && python3 -c "import sys;sys.exit(0 if $d <= 0.5 else 1)" && [ $retries -lt 2 ]; then
    # aborted by a safety check before the jaw ever closed, dish untouched: not an attempt - retry it
    retries=$((retries + 1)); echo "run $k: aborted before any grip, dish untouched - retry ($retries/2), streak stays $ok"
  else
    echo "run $k: FAILED - stopping"; break
  fi
  grep -qE "control loop died|fail to communicate|Motor error" $L && { echo "stopping: arm/driver error"; break; }
done
echo "=== finished: $ok/$N in a row ($k runs, $retries retried safety aborts)"
