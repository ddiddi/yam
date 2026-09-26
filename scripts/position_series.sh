#!/bin/zsh
# ./scripts/position_series.sh START_XY "x,y x,y ...": move one object through a list of positions - each run picks it
# where the last run set it down and places it at the next target (pick_place.py --place-dist/--place-dir). After
# each run cam1 re-locates the object's top (LID_Z, default 0.12 m) for a check. Stops at the first failure or
# captures/live/STOP. Logs -> captures/live/series_<k>.log, summary -> captures/live/series.log
cd "$(dirname "$0")/.."
export YAM_CAM_MAP=${YAM_CAM_MAP:-"0:0,1:1,2:3,3:2"}
cur=$1; TARGETS=(${=2})
D=${D:-0.055}; H=${H:-0.12}; Z=${Z:-0.04}; LOAD=${LOAD:-0.28}; SPEED=${SPEED:-1}; HOLD=${HOLD:-1}
OBST=${OBST:-"0.23,-0.11,0.09,0.06;0.30,-0.11,0.09,0.06;0.37,-0.11,0.09,0.06"}  # the scale, as three posts
rm -f captures/live/STOP
k=0
for tgt in $TARGETS; do
  k=$((k + 1))
  [ -f captures/live/STOP ] && { echo "stopping: STOP file"; break; }
  .venv/bin/python calib_tools/cam_rebump.py --cam cam1 --write >/dev/null 2>&1
  read dist dir <<< $(python3 -c "import math;a=[float(v) for v in '$cur'.split(',')];b=[float(v) for v in '$tgt'.split(',')];print(f'{math.dist(a,b):.4f} {math.degrees(math.atan2(b[1]-a[1],b[0]-a[0])):.1f}')")
  echo "=== move $k: ($cur) -> ($tgt): $dist m at $dir deg"
  L=captures/live/series_$k.log
  YAM_BATCH="position series $k/${#TARGETS}" .venv/bin/python -u pick_place.py --run --speed $SPEED --palm --grasp side --side-only --side-tilt 75 \
    --side-cross-max 1.0 --z-grasp $Z --soft --grip-load $LOAD --grip-load-max 0.45 --hold $HOLD --place-dist $dist --place-dir $dir \
    --no-servo --fast --cam-moved-px 8 --margin 0.015 --tactile --given-only --given "$cur,$D,$H;$OBST" \
    --record datasets/yam_positions --task "${TASK:-pick up the object and set it down at a new spot}" > $L 2>&1
  st=$(python3 -c "import json;print(json.load(open('captures/live/state.json'))['status'])")
  grep -E "grip check|hold test|still holding at|!!" $L | grep -v "has moved\|tactile skin not"
  placed=$(grep -o "place (+[0-9.-]*, [+-][0-9.]*)" $L | tail -1 | grep -o "[+-][0-9.]*" | tr -d '+' | paste -sd, -)
  if [ "$st" = "done" ] && grep -q "still holding at the place point" $L; then
    cur=${placed:-$tgt}
    echo "move $k: OK - set down at ($cur)"
  else
    echo "move $k: FAILED (status $st) - stopping"; break
  fi
done
echo "=== series finished after $k move(s); the object is at ($cur)"
