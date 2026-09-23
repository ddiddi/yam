"""Between runs of a swap batch: wait until a person has put a NEW object in the zone and stepped away.

    python wait_new_object.py --batch "3/10" [--timeout 900]     # camera only; exit 0 = go, 1 = timed out

The arm is parked at rest. The reference is the zone as the last run left it. We wait for the zone to
change (the object swapped), then for it to be completely still for STILL_S seconds (hands out of the
zone and its surroundings), then for the survey + depth analysis to find an object, then count down.
Every stage is written to the live map's state, so the person sees what the arm is waiting for.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
STATE = HERE / "captures" / "live" / "state.json"
CHANGED = 0.01  # fraction of zone pixels that must differ from the reference: the object was swapped
STILL_PX = 0.004  # fraction of watch-window pixels allowed to move between frames and still be "still"
STILL_S = 4.0  # seconds of stillness before the arm may move
COUNTDOWN = 5


def say(batch: str, phase: str, status: str = "planning") -> None:
    try:
        d = json.loads(STATE.read_text()) if STATE.exists() else {}
    except Exception:
        d = {}
    d.update(status=status, phase=phase, batch=batch, steps=[], step=-1, path=[], trail=[],
             updated=int(time.time() * 1000))
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d))
    tmp.replace(STATE)
    print(f"[{time.strftime('%H:%M:%S')}] {phase}", flush=True)


def main() -> int:
    import pick_place as pp

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="")
    ap.add_argument("--timeout", type=float, default=900.0)
    a = ap.parse_args()
    cams = pp.load_cameras()
    cam = cams[pp.DET]
    zone = pp.load_zone()
    def frame() -> np.ndarray:
        # a fresh open per look: one capture held open for minutes went on serving the same stale frame,
        # and a cup put in the zone was never seen
        return pp.grab(int(pp.DET[3:]))

    f0 = frame()
    poly = cam.project(np.c_[zone, np.zeros(len(zone))]).astype(np.int32)
    zmask = np.zeros(f0.shape[:2], np.uint8)
    cv2.fillPoly(zmask, [poly], 255)
    # watch the zone plus a wide margin around it: a hand reaching in moves there first
    watch = cv2.dilate(zmask, np.ones((121, 121), np.uint8))
    g = lambda f: cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    ref = g(f0)
    t0 = time.time()
    say(a.batch, f"run {a.batch}: waiting for you to swap the object (arm parked)")
    # 1. the zone changes
    while True:
        if time.time() - t0 > a.timeout:
            say(a.batch, "stopped: no new object within the time limit", "done")
            return 1
        cur = g(frame())
        if (np.abs(cur.astype(int) - ref)[zmask > 0] > 25).mean() > CHANGED:
            break
        time.sleep(0.5)
    # 2. then it is still for STILL_S (hands gone), and 3. an object is there
    while True:
        say(a.batch, f"run {a.batch}: object change seen - waiting for hands to leave the zone")
        prev, still_since = g(frame()), time.time()
        while time.time() - still_since < STILL_S:
            if time.time() - t0 > a.timeout:
                say(a.batch, "stopped: the zone never became still", "done")
                return 1
            time.sleep(0.4)
            cur = g(frame())
            if (np.abs(cur.astype(int) - prev)[watch > 0] > 20).mean() > STILL_PX:
                still_since = time.time()  # something moved: start the still window again
            prev = cur
        f = frame()
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            objs = pp.survey(f, cam, 2000.0, quiet=True)
            objs = pp.fuse_depth(objs, f, cam, more=[frame() for _ in range(pp.DEPTH_FRAMES - 1)])
        if objs:
            desc = "; ".join(f"{o.name} {o.grip_width * 100:.1f} cm wide x {o.height * 100:.1f} cm" for o in objs)
            print(f"   found: {desc}", flush=True)
            break
        say(a.batch, f"run {a.batch}: the zone is still but empty - place an object")
        ref = g(f)
        while (np.abs(g(frame()).astype(int) - ref)[zmask > 0] > 25).mean() <= CHANGED:
            if time.time() - t0 > a.timeout:
                say(a.batch, "stopped: no object within the time limit", "done")
                return 1
            time.sleep(0.5)
    for s in range(COUNTDOWN, 0, -1):
        say(a.batch, f"run {a.batch}: {len(objs)} object(s) found - hands clear, the arm moves in {s} s")
        time.sleep(1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
