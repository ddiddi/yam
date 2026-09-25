#!/bin/zsh
# ./scripts/e2e.sh [OZ] [dish_guess_xy] [bottle_guess_xy]: the end-to-end experiment in one go -
#   1. the Petri dish from the tape strip onto the scale, weighed (pick_place.py --station --station-stage on)
#   2. the wash bottle picked low, tilted over the dish, spout aimed with the detection camera, OZ squeeze-poured
#      (read on the scale's display), bottle put back
#   3. the dish off the scale and back where it was (--station-stage off)
# Before every stage the detection camera is checked against its background (a small pure turn is re-aimed with
# calib_tools/cam_rebump.py, anything else stops) and the objects it handles are re-measured from a fresh frame.
# Stops at the first failed stage. Logs -> captures/live/e2e_<stage>.log
cd "$(dirname "$0")/.."
export YAM_CAM_MAP=${YAM_CAM_MAP:-"0:0,1:1,2:3,3:2"}
OZ=${1:-0.2}; DG=${2:-0.27,0.08}; BG=${3:-0.355,0.12}
PLACE=${PLACE:-0.31,-0.12}              # where the dish goes on the scale (reachable, 5 cm inside its edge)
TIP=${TIP:--0.010,-0.064,0.20}          # the spout tip from the bottle axis (spout turned toward the scale)
TILT=${TILT:-60}                        # a shallower tilt than 75: the full bottle dripped when tipped further
OFF_X=0.009
MARGIN=${MARGIN:-0.015}                 # stage 1/3 planner margin (m)
SLACK=${SLACK:-0.004}                   # --table-slack for the low dish legs
SPOUT_POST=${SPOUT_POST:-1}             # 0: do not model the spout as a post (only the bottle body)
RESCAN=${RESCAN:-}                      # "--no-rescan": skip the mid-run re-look (its 3-camera carve can inflate the bottle)
rm -f captures/live/STOP

check_cam() {  # 0 = fine (re-aimed if it had turned a little), 1 = stop
  local rep=$(.venv/bin/python calib_tools/cam_rebump.py --cam cam1 2>/dev/null | grep "^cam1")
  echo "   camera: $rep"
  local turn=$(echo $rep | grep -o "turn of [0-9.]*" | grep -o "[0-9.]*$")
  local good=$(echo $rep | grep -o "^cam1: [0-9]*/[0-9]*" | cut -d' ' -f2)
  python3 - "$turn" "$good" <<'PY'
import sys
t = float(sys.argv[1] or 99); a, b = (int(v) for v in (sys.argv[2] or "0/1").split("/"))
sys.exit(0 if t < 0.05 else (2 if (t < 1.5 and a / b > 0.6) else 1))  # objects moved in the view lower the share
PY
  local rc=$?
  [ $rc -eq 0 ] && return 0
  [ $rc -eq 1 ] && return 1
  .venv/bin/python calib_tools/cam_rebump.py --cam cam1 --write >/dev/null 2>&1 && echo "   camera re-aimed" && return 0
  return 1
}
state() { python3 -c "import json;print(json.load(open('captures/live/state.json'))['status'])"; }

START=${START:-1}                       # 2: the dish is already on the scale - start at the pour; 3: at taking it back
if [ $START -le 1 ]; then
# ---- 1. dish onto the scale
check_cam; [ $? -ne 0 ] && { echo "!! the detection camera moved more than a small turn: recalibrate"; exit 1; }
# DISH_XY / BOTTLE_XY: positions measured by hand-checked fits (skip the automatic finders, which need the object's
# base on bare blue tape); the arm puts the bottle back where it picked it, so it stays valid through the stages
dish=${DISH_XY:-$(.venv/bin/python scripts/dish_find_rim.py $DG 2>/dev/null | cut -d, -f1,2)}
bottle=${BOTTLE_XY:-$(.venv/bin/python scripts/bottle_find.py $BG 2>/dev/null | cut -d, -f1,2)}
[ -z "$dish" ] && { echo "!! dish not found near $DG"; exit 1; }
[ -z "$bottle" ] && { echo "!! bottle not found near $BG"; exit 1; }
gap=$(python3 -c "import math;a=[float(v) for v in '$dish'.split(',')];b=[float(v) for v in '$bottle'.split(',')];print(f'{math.dist(a,b)*100:.1f}')")
spout=$(python3 -c "b=[float(v) for v in '$bottle'.split(',')];t=[float(v) for v in '$TIP'.split(',')];print(f'{b[0]+t[0]:.4f},{b[1]+t[1]:.4f}')")
post=""; [ "$SPOUT_POST" = "1" ] && post=";$spout,0.03,0.25"
echo "=== dish at ($dish), bottle at ($bottle), $gap cm apart; spout tip ~($spout)"
python3 -c "import sys;sys.exit(0 if $gap >= 11.0 else 1)" || { echo "!! the dish and bottle are only $gap cm apart (need 11): move them apart"; exit 1; }
YAM_BATCH="e2e 1/3: dish onto the scale" .venv/bin/python -u pick_place.py --run --station --station-stage on --station-place $PLACE \
  --weigh 8 --object 0 --grasp side --side-only --side-cross-max 1.0 --side-tilt 42 40 --z-grasp 0.013 --max-width 0.092 \
  --soft --grip-load 0.18 --margin $MARGIN --table-slack $SLACK --no-servo --cam-moved-px 8 --tactile ${=RESCAN} \
  --given-only --given "$dish,0.09,0.015;$bottle,0.0785,0.25$post" --record datasets/yam_e2e \
  --task "pick up the petri dish and set it on the scale" > captures/live/e2e_1.log 2>&1
grep -E "grip check|hold test|still holding|scale display|!!" captures/live/e2e_1.log | grep -v "has moved\|tactile skin not"
[ "$(state)" = "done" ] && grep -q "still holding at the place point" captures/live/e2e_1.log || { echo "!! stage 1 failed"; exit 1; }
w=$(.venv/bin/python -c "import cv2,scale_read,glob;fs=sorted(glob.glob('captures/weigh_cam2_*.jpg'));r=scale_read.read(cv2.imread(fs[-1])) if fs else None;print('?' if r is None else f'{r[0]:.1f}')")
echo "=== stage 1 done: the dish weighs $w oz"
else
  dish=${DISH_XY:?DISH_XY (where the dish goes back) is needed with START>1}; bottle=${BOTTLE_XY:?BOTTLE_XY is needed with START>1}
  spout=$(python3 -c "b=[float(v) for v in '$bottle'.split(',')];t=[float(v) for v in '$TIP'.split(',')];print(f'{b[0]+t[0]:.4f},{b[1]+t[1]:.4f}')")
fi
if [ $START -le 2 ]; then

# ---- 2. pour into the dish
check_cam; [ $? -ne 0 ] && { echo "!! the detection camera moved: stopping before the pour"; exit 1; }
[ -z "$BOTTLE_XY" ] && bottle=$(.venv/bin/python scripts/bottle_find.py $bottle 2>/dev/null | cut -d, -f1,2)
[ -z "$bottle" ] && { echo "!! bottle not found"; exit 1; }
aim=$(python3 -c "x,y='$bottle'.split(',');print(f'{float(x)+$OFF_X:.4f},{float(y):.4f}')")
px=${PLACE%,*}; py=${PLACE#*,}
YAM_BATCH="e2e 2/3: pour $OZ oz into the dish" .venv/bin/python -u pick_place.py --run --station --palm --grasp side --side-only \
  --side-tilt 75 --side-cross-max 1.0 --z-grasp 0.06 --soft --grip-load 0.50 --grip-load-max 0.75 --grip-force 80 \
  --skin-max 99999 --no-servo --fast --cam-moved-px 8 --margin 0.015 --tactile --pour-oz $OZ --pour-at $PLACE \
  --pour-tip $TIP --pour-base-z 0.14 --pour-tilt-deg $TILT --pour-lip 0.12 --pour-bowl $px,$py,0.045,0.067 \
  --pour-load-max 99 --pour-max-squeeze 0.030 --given-only --given "$aim,0.0785,0.25;$PLACE,0.09,0.07" \
  --record datasets/yam_e2e --task "tilt the wash bottle over the dish on the scale and squeeze-pour $OZ oz" > captures/live/e2e_2.log 2>&1
grep -E "grip check|hold test|spout aim|scale before|flowing|pour done|after the pour|still holding|!!" captures/live/e2e_2.log | grep -v "has moved\|tactile skin not"
[ "$(state)" = "done" ] || { echo "!! stage 2 failed"; exit 1; }
grep -q "pour done" captures/live/e2e_2.log || { echo "!! stage 2 did not pour (see the spout aim above): stopping"; exit 1; }
fi

# ---- 3. the dish back to its spot
check_cam; [ $? -ne 0 ] && { echo "!! the detection camera moved: stopping before taking the dish back"; exit 1; }
[ -z "$BOTTLE_XY" ] && bottle=$(.venv/bin/python scripts/bottle_find.py $bottle 2>/dev/null | cut -d, -f1,2)
spout=$(python3 -c "b=[float(v) for v in '$bottle'.split(',')];t=[float(v) for v in '$TIP'.split(',')];print(f'{b[0]+t[0]:.4f},{b[1]+t[1]:.4f}')")
post=""; [ "$SPOUT_POST" = "1" ] && post=";$spout,0.03,0.25"
YAM_BATCH="e2e 3/3: dish back from the scale" .venv/bin/python -u pick_place.py --run --station --station-stage off --station-place $PLACE \
  --weigh 8 --object 0 --grasp side --side-only --side-cross-max 1.0 --side-tilt 42 40 --z-grasp 0.013 --max-width 0.092 \
  --soft --grip-load 0.18 --margin $MARGIN --table-slack $SLACK --no-servo --fast --cam-moved-px 8 --tactile \
  --given-only --given "$dish,0.09,0.015;$bottle,0.0785,0.25$post" --record datasets/yam_e2e \
  --task "take the petri dish off the scale and put it back" > captures/live/e2e_3.log 2>&1
grep -E "grip check|hold test|still holding|!!" captures/live/e2e_3.log | grep -v "has moved\|tactile skin not"
[ "$(state)" = "done" ] || { echo "!! stage 3 failed"; exit 1; }
echo "=== all three stages done"
