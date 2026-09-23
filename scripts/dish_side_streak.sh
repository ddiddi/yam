#!/bin/zsh
# load-grip side pick of the dish until 2 succeed (max 6 attempts); re-measure the dish before each
cd /Users/solotech/yam
ok=0; n=0
while [ $ok -lt 5 ] && [ $n -lt 5 ]; do
  n=$((n+1))
  [ -f captures/live/STOP ] && { echo "stopping: STOP file"; exit 1; }
  xy=$(.venv/bin/python - <<'PY' 2>/dev/null
import pick_place as pp
from scene3d import grab
class LC:
    def __init__(s, f): s.f = f
    def latest(s): return s.f
pp.FUSED_CAMS[pp.DET] = LC(grab(1)); pp.FUSED_MODELS.update(pp.load_cameras())
c = pp.measure_disc(0.045, 0.015)
print("" if c is None else f"{c[0]:.3f},{c[1]:.3f}")
PY
)
  echo "=== attempt $n (streak $ok/5): dish at ($xy)"
  [ -z "$xy" ] && { echo "dish not found - stopping"; exit 1; }
  L=captures/live/run_streak_$n.log
  YAM_BATCH="dish load-grip streak $n/5" .venv/bin/python -u pick_place.py --run --object 0 --grasp side --side-only \
    --side-cross-max 1.0 --side-tilt 42 40 --z-grasp 0.013 --max-width 0.092 --soft --no-servo --shuttle \
    --given-only --given "$xy,0.09,0.015" --record datasets/yam_zone_swap --tactile > $L 2>&1
  st=$(python3 -c "import json;print(json.load(open('captures/live/state.json'))['status'])")
  grep -E "soft contact|grip check|hold test|fingertips at" $L | tail -4
  if [ "$st" = "done" ] && grep -q "OK, lifting" $L; then ok=$((ok+1)); echo "attempt $n: SUCCESS ($ok/5 in a row)"; else echo "attempt $n: FAILED ($st) - streak broken"; exit 2; fi
  grep -qE "control loop died|fail to communicate|Motor error" $L && { echo "stopping: arm/driver error"; exit 1; }
done
echo "=== finished: $ok/5 in a row in $n attempts"
