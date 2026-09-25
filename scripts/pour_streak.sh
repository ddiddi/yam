#!/bin/zsh
# ./scripts/pour_streak.sh [N] [oz] [x,y]: squeeze-pour `oz` into the bowl on the scale N times in a row (default 5 x 0.5 oz)
# with the full-depth palm grip low on the body (Z, default 6 cm), 75 deg tilt and step-and-settle dosing of pick_place.py --pour-oz.
# Before each run the bottle is re-found (scripts/bottle_find.py, near where it was last). A run counts if it finished
# ("done"), poured within 0.15 oz of the target and the bottle is back within 2 cm of where it was picked. Stops at
# the first failure, an arm/driver error, or captures/live/STOP. Logs -> captures/live/pour_streak_<k>.log
cd "$(dirname "$0")/.."
export YAM_CAM_MAP="1:2,2:1"
N=${1:-5}; OZ=${2:-0.5}; xy=${3:-0.332,0.136}; Z=${Z:-0.06}  # Z: fingertip height of the grip (low on the body squeezes best)
OFF_X=0.009  # the arm's offset from the camera fit (as in the bottle runs)
rm -f captures/live/STOP
k=0; ok=0
while [ $ok -lt $N ]; do
  k=$((k + 1))
  [ -f captures/live/STOP ] && { echo "stopping: STOP file"; break; }
  new=""
  for g in "$xy" "$(python3 -c "x,y='$xy'.split(',');print(f'{float(x)-0.03:.3f},{float(y)-0.015:.3f}')")"; do
    new=$(.venv/bin/python scripts/bottle_find.py "$g" 2>/dev/null)
    [ -n "$new" ] && break
  done
  [ -z "$new" ] && { echo "run $k: bottle not found near ($xy)"; break; }
  xy=$(echo $new | cut -d, -f1,2)
  aim=$(python3 -c "x,y='$xy'.split(',');print(f'{float(x)+$OFF_X:.4f},{float(y):.4f}')")
  echo "=== run $k (streak $ok/$N): bottle at ($xy), aiming ($aim), pouring $OZ oz"
  L=captures/live/pour_streak_$k.log
  YAM_BATCH="pour $OZ oz $((ok + 1))/$N" .venv/bin/python -u pick_place.py --run --station --palm --grasp side --side-only \
    --side-tilt 75 --side-cross-max 1.0 --z-grasp $Z --soft --grip-load 0.50 --grip-load-max 0.75 --grip-force 80 \
    --skin-max 99999 --no-servo --fast --cam-moved-px 15 --margin 0.015 --tactile --refind scripts/bottle_find.py \
    --refind-offset $OFF_X,0 --pour-oz $OZ --pour-at 0.313,-0.10 --pour-tip -0.010,-0.064,0.20 --pour-base-z 0.14 \
    --pour-tilt-deg 75 --pour-lip 0.16 --pour-bowl 0.313,-0.123,0.08,0.117 --pour-load-max 99 --pour-max-squeeze 0.030 \
    --given-only --given "$aim,0.0785,0.25;0.313,-0.123,0.16,0.117" --record datasets/yam_squeeze_pour \
    --task "tilt the wash bottle toward the bowl and squeeze-pour $OZ oz" > $L 2>&1
  st=$(python3 -c "import json;print(json.load(open('captures/live/state.json'))['status'])")
  grep -E "grip check|flowing|pour done|after the pour|still holding at|!!" $L | grep -v "has moved\|tactile skin not\|bottle tilt:"
  after=""
  for try in 1 2 3; do  # the arm may still be folding away through the view: wait, look again
    sleep 2
    after=$(.venv/bin/python scripts/bottle_find.py "$xy" 2>/dev/null | cut -d, -f1,2)
    [ -n "$after" ] && break
  done
  d="-"
  [ -n "$after" ] && d=$(python3 -c "import math;a=[float(v) for v in '$xy'.split(',')];b=[float(v) for v in '$after'.split(',')];print(f'{math.dist(a,b)*100:.1f}')")
  poured=$(grep -o "poured [0-9.]* of" $L | tail -1)
  echo "run $k: status $st, $poured $OZ oz, bottle back at ($after), $d cm from the pick point"
  p=$(echo "$poured" | grep -o "[0-9.]*" | head -1)
  good=$(python3 -c "print(1 if '$p' and abs(float('$p') - $OZ) <= 0.15 else 0)")
  if [ "$st" = "done" ] && [ "$good" = "1" ] && [ "$d" != "-" ] && python3 -c "import sys;sys.exit(0 if $d <= 2.0 else 1)"; then
    ok=$((ok + 1)); echo "run $k: SUCCESS ($ok in a row)"; xy=$after
  else
    echo "run $k: FAILED - stopping"; break
  fi
  grep -qE "control loop died|fail to communicate|Motor error" $L && { echo "stopping: arm/driver error"; break; }
done
echo "=== finished: $ok/$N in a row ($k runs)"
