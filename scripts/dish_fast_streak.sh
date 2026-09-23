#!/bin/zsh
# 5 dish picks in a row at 2x speed with blended legs; the dish is located by colour before each; stops at the first failure
cd /Users/solotech/yam
rm -f captures/live/STOP
for n in 1 2 3 4 5; do
  [ -f captures/live/STOP ] && { echo "stopping: STOP file"; exit 1; }
  xy=$(.venv/bin/python scripts/dish_find.py 2>/dev/null)
  echo "=== run $n/5: dish at ($xy)"
  [ -z "$xy" ] && { echo "dish not found - stopping"; exit 1; }
  L=captures/live/run_dish_fast_$n.log
  YAM_BATCH="dish fast $n/5" .venv/bin/python -u pick_place.py --run --speed 2 --blend --hold 0 --no-rescan --cam-moved-px 8 \
    --object 0 --grasp side --side-only --side-cross-max 1.0 --side-tilt 42 40 --z-grasp 0.013 --max-width 0.092 --soft \
    --no-servo --given-only --given "$xy,0.09,0.015" --record datasets/yam_zone_swap --tactile > $L 2>&1
  st=$(python3 -c "import json;print(json.load(open('captures/live/state.json'))['status'])")
  grep -E "soft contact|grip check|hold test|fingertips at|saved LeRobot" $L | tail -5
  # did the dish really end up at the place point? locate it again and compare
  place=$(grep -oE "place \(\+?-?[0-9.]+, \+?-?[0-9.]+\)" $L | tail -1 | grep -oE "[-0-9.]+, [-+0-9.]+" | tr -d ' +')
  now=$(.venv/bin/python scripts/dish_find.py 2>/dev/null)
  off=$(python3 -c "import math;a=[float(v) for v in '$place'.split(',')];b=[float(v) for v in '$now'.split(',')];print(f'{100*math.dist(a,b):.1f}')" 2>/dev/null)
  echo "   planned place ($place), dish found at ($now): ${off:-?} cm off"
  if [ "$st" = "done" ] && grep -q "OK, lifting" $L && [ -n "$off" ] && python3 -c "import sys;sys.exit(0 if $off <= 2.5 else 1)"; then
    echo "run $n: SUCCESS ($n/5 in a row)"
  else echo "run $n: FAILED ($st, dish ${off:-?} cm from the place point)"; exit 2; fi
  grep -qE "control loop died|fail to communicate|Motor error" $L && { echo "stopping: arm/driver error"; exit 1; }
done
echo "=== finished: 5/5 in a row"
