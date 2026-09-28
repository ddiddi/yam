#!/bin/zsh
# ./scripts/position_series.sh START_XY "x,y x,y ...": move one object through a list of positions - each run picks it
# where the last run set it down and places it at the next target (pick_place.py --place-dist/--place-dir). After
# each run cam1 re-locates the object's top (LID_Z, default 0.12 m) for a check. Stops at the first failure or
# captures/live/STOP. Logs -> captures/live/series_<k>.log, summary -> captures/live/series.log
cd "$(dirname "$0")/.."
export YAM_CAM_MAP=${YAM_CAM_MAP:-"0:1,1:0"}
cur=$1; TARGETS=(${=2})
D=${D:-0.055}; H=${H:-0.12}; Z=${Z:-0.04}; LOAD=${LOAD:-0.28}; SPEED=${SPEED:-1}; HOLD=${HOLD:-1}
OBST=${OBST-}  # extra given obstacles ("x,y,d,h;..."), none by default (the 2026-09-26 setup has no scale)
FIND=${FIND:-1}  # 1: re-find the object with scripts/cyl_find.py before every pick and after every set-down
GLMAX=${GLMAX:-0.45}; SLACK=${SLACK:-0.004}; TILT=${TILT:-75}; PAUSE=${PAUSE:-0}  # PAUSE: s of rest between moves
# the camera re-exposes for a few seconds after the arm leaves its view: retry the finder before giving up
findit() { local r; for i in 1 2 3 4 5; do r=$(.venv/bin/python ${FINDER:-scripts/sticker_find.py} $1 $D $H 2>/dev/null); [ -n "$r" ] && { echo $r; return; }; sleep 3; done; }
rm -f captures/live/STOP
k=0
for tgt in $TARGETS; do
  k=$((k + 1))
  [ -f captures/live/STOP ] && { echo "stopping: STOP file"; break; }
  [ $k -gt 1 ] && [ $PAUSE -gt 0 ] && sleep $PAUSE
  # cam1's auto-exposure blows out the pale sticker on the white sheet: fixed manual exposure (C270 at USB location
  # 0x00110000 = cam1; tools/uvc-util, built from github.com/jtfrey/uvc-util). Re-applied every move (lost on replug)
  [ -x tools/uvc-util ] && tools/uvc-util -L ${CAM1_LOC:-0x00110000} -s auto-exposure-mode=1 -s exposure-time-abs=${EXPO:-140} >/dev/null 2>&1
  for c in cam0 cam1; do .venv/bin/python calib_tools/cam_rebump.py --cam $c --write >/dev/null 2>&1; done  # both creep
  if [ "$FIND" = "1" ]; then
    f=$(findit $cur)
    [ -z "$f" ] && { echo "move $k: the camera cannot find the object near ($cur) - stopping"; break; }
    fx=$(echo $f | cut -d, -f1,2); off=$(python3 -c "import math;a=[float(v) for v in '$cur'.split(',')];b=[float(v) for v in '$fx'.split(',')];print(f'{math.dist(a,b)*100:.1f}')")
    echo "   re-found at ($fx), $off cm from where it should be (score ${f##*,})"
    python3 -c "import sys;sys.exit(0 if $off <= 3.0 else 1)" || { echo "move $k: object $off cm from where it should be - stopping"; break; }
    cur=$fx
  fi
  read dist dir <<< $(python3 -c "import math;a=[float(v) for v in '$cur'.split(',')];b=[float(v) for v in '$tgt'.split(',')];print(f'{math.dist(a,b):.4f} {math.degrees(math.atan2(b[1]-a[1],b[0]-a[0])):.1f}')")
  echo "=== move $k: ($cur) -> ($tgt): $dist m at $dir deg"
  L=captures/live/series_$k.log
  YAM_BATCH="position series $k/${#TARGETS}" .venv/bin/python -u pick_place.py --run --speed $SPEED --palm --grasp side --side-only --side-tilt $TILT \
    --side-cross-max 1.0 --z-grasp $Z --soft --grip-load $LOAD --grip-load-max $GLMAX --table-slack $SLACK --hold $HOLD --place-dist $dist --place-dir $dir \
    --no-servo --fast --cam-moved-px 8 --margin 0.015 --tactile --given-only --given "$cur,$D,$H${OBST:+;$OBST}" \
    --record datasets/yam_positions --task "${TASK:-pick up the object and set it down at a new spot}" > $L 2>&1
  st=$(python3 -c "import json;print(json.load(open('captures/live/state.json'))['status'])")
  grep -E "grip check|hold test|still holding at|!!" $L | grep -v "has moved\|tactile skin not"
  placed=$(grep -o "place (+[0-9.-]*, [+-][0-9.]*)" $L | tail -1 | grep -o "[+-][0-9.]*" | tr -d '+' | paste -sd, -)
  # a perfect run only: no grasp retry, no warning, no motor fault
  if grep -vE "cam[0-9] has moved .* left out of the fused workspace" $L | grep -qE "closed on nothing|attempt [0-9]/|!!|Motor error|loss communication|control loop died"; then
    grep -q "still holding at the place point" $L && cur=${placed:-$tgt}  # it was still set down: report where
    echo "move $k: not clean (retry, warning or motor fault) - stopping"; break
  fi
  if [ "$st" = "done" ] && grep -q "still holding at the place point" $L; then
    cur=${placed:-$tgt}
    if [ "$FIND" = "1" ]; then  # check the set-down with the camera (the arm is folded out of the view by now)
      f=$(findit $cur)
      [ -z "$f" ] && { echo "move $k: set down, but the camera cannot find it near ($cur) - stopping"; break; }
      fx=$(echo $f | cut -d, -f1,2); off=$(python3 -c "import math;a=[float(v) for v in '$cur'.split(',')];b=[float(v) for v in '$fx'.split(',')];print(f'{math.dist(a,b)*100:.1f}')")
      echo "   set-down check: at ($fx), $off cm from the target"
      python3 -c "import sys;sys.exit(0 if $off <= 3.0 else 1)" || { echo "move $k: set down $off cm off target - stopping"; break; }
    fi
    echo "move $k: OK - set down at ($cur)"
  else
    echo "move $k: FAILED (status $st) - stopping"; break
  fi
done
echo "=== series finished after $k move(s); the object is at ($cur)"
