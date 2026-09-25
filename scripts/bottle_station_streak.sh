#!/bin/zsh
# ./scripts/bottle_station_streak.sh [N] [x,y]: the water-filled wash bottle weigh-station cycle N times in a row (default 5),
# with the full-depth palm grip of trajectories/wash_bottle_palm_weigh_2026-09-24.json.
# Before each run the bottle is re-found (scripts/bottle_find.py, near where it was last) and the empty scale photographed
# (the tare); a run counts only if it finished, both grips passed their check, and the bottle is back within 2 cm of where
# it was picked. Stops at the first failure, an arm/driver error, or captures/live/STOP. At most 2 aborts before any grip
# (bottle untouched) are retried. Weigh photos -> captures/weigh_bottle_run<k>_cam*_*.jpg, tares -> captures/weigh_bottle_tare<k>.jpg
cd "$(dirname "$0")/.."
export YAM_CAM_MAP="1:2,2:1"
N=${1:-5}; xy=${2:-0.2693,0.0738}
OFF_X=0.009  # the arm's offset from the camera fit (same as the successful run)
rm -f captures/live/STOP
ok=0; k=0; retries=0
while [ $ok -lt $N ]; do
  k=$((k + 1))
  [ -f captures/live/STOP ] && { echo "stopping: STOP file"; break; }
  new=$(.venv/bin/python scripts/bottle_find.py "$xy" 2>captures/live/bottle_find_$k.err)
  [ -z "$new" ] && { echo "run $k: bottle not found near ($xy): $(cat captures/live/bottle_find_$k.err)"; break; }
  xy=$(echo $new | cut -d, -f1,2)
  aim=$(python3 -c "x,y='$xy'.split(',');print(f'{float(x)+$OFF_X:.4f},{float(y):.4f}')")
  .venv/bin/python -c "import cv2; from scene3d import grab; cv2.imwrite('captures/weigh_bottle_tare$k.jpg', grab(2))" 2>/dev/null
  echo "=== run $k (streak $ok/$N): bottle at ($xy), aiming ($aim)"
  L=captures/live/station_bottle_streak_$k.log
  YAM_BATCH="wash bottle weigh $((ok + 1))/$N" .venv/bin/python -u pick_place.py --run --station --palm --weigh 8 --object 0 \
    --grasp side --side-only --side-tilt 75 --side-cross-max 1.0 --z-grasp 0.08 --soft --grip-load 0.45 --grip-load-max 0.65 \
    --grip-force 50 --skin-max 5000 --no-servo --tactile --refind scripts/bottle_find.py --refind-offset $OFF_X,0 \
    --given-only --given "$aim,0.0785,0.22" --record datasets/yam_weigh_station \
    --task "pick up the water-filled wash bottle at full depth, set it on the scale, then put it back where it was" > $L 2>&1
  st=$(python3 -c "import json;print(json.load(open('captures/live/state.json'))['status'])")
  for f in captures/weigh_cam*_*.jpg(N); do mv $f captures/weigh_bottle_run${k}_${f:t:r:s/weigh_//}.jpg; done
  grips=$(grep -c "OK.*lifting" $L)
  grep -E "soft contact|grip check|hold test|still holding|scale display|re-measured|!!" $L | grep -v "has moved\|tactile skin not" | tail -8
  after=$(.venv/bin/python scripts/bottle_find.py "$xy" 2>/dev/null | cut -d, -f1,2)
  d=$([ -n "$after" ] && python3 -c "import math;a=[float(v) for v in '$xy'.split(',')];b=[float(v) for v in '$after'.split(',')];print(f'{math.dist(a,b)*100:.1f}')" || echo "-")
  echo "run $k: status $st, grips passed $grips (2 needed), bottle back at ($after), $d cm from the pick point"
  if [ "$st" = "done" ] && [ "$grips" -ge 2 ] && [ "$d" != "-" ] && python3 -c "import sys;sys.exit(0 if $d <= 2.0 else 1)"; then
    ok=$((ok + 1)); echo "run $k: SUCCESS ($ok in a row)"; xy=$after
  elif [ "$grips" = "0" ] && [ "$d" != "-" ] && python3 -c "import sys;sys.exit(0 if $d <= 0.5 else 1)" && [ $retries -lt 2 ]; then
    retries=$((retries + 1)); echo "run $k: aborted before any grip, bottle untouched - retry ($retries/2), streak stays $ok"
  else
    echo "run $k: FAILED - stopping"; break
  fi
  grep -qE "control loop died|fail to communicate|Motor error" $L && { echo "stopping: arm/driver error"; break; }
done
echo "=== finished: $ok/$N in a row ($k runs, $retries retried safety aborts)"
