"""Is each camera pointed at the workspace? Per device: share of the frame that is blue tape, where the tape sits in
the frame, and how many ChArUco sheet markers it reads - with a verdict and what to change.

    python calib_tools/aim_check.py            # every device OpenCV opens -> captures/aim_check.jpg
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

Y = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Y / "calib_tools"))
import station_calib as sc  # noqa: E402

b, mdet, _ = sc.board(0.75)
tiles = []
for i in range(6):
    c = cv2.VideoCapture(i)
    if not c.isOpened():
        continue
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    for _ in range(15):
        ok, f = c.read()
        time.sleep(0.03)
    c.release()
    if not ok:
        continue
    h, w = f.shape[:2]
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    blue = ((hsv[..., 0] > 95) & (hsv[..., 0] < 125) & (hsv[..., 1] > 90)).astype(np.uint8)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    frac = blue.mean()
    _, ids, _ = mdet.detectMarkers(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    n = 0 if ids is None else len(ids)
    tips = []
    if frac > 0.02:
        ys, xs = np.nonzero(blue)
        cx, cy = xs.mean() / w, ys.mean() / h
        if cy < 0.35:
            tips.append("tilt DOWN (the tape is in the top of the frame)")
        if cy > 0.8:
            tips.append("tilt UP a little (the tape is at the bottom edge)")
        if cx < 0.25:
            tips.append("turn LEFT (the tape is at the left edge)")
        if cx > 0.75:
            tips.append("turn RIGHT (the tape is at the right edge)")
    else:
        tips.append("does not see the workspace - point it at the blue tape")
    if frac > 0.02 and frac < 0.12:
        tips.append("move it closer or aim more squarely: the tape fills only %.0f%% of the view" % (frac * 100))
    good = frac >= 0.12 and n >= 8 and len(tips) == 0
    verdict = "GOOD" if good else ("OK" if frac >= 0.12 else "RE-AIM")
    print(f"device {i}: tape {frac * 100:4.1f}% of the frame, {n:2d} sheet markers -> {verdict}"
          + ("" if not tips else ": " + "; ".join(tips)))
    x = cv2.resize(f, (640, 360))
    cv2.putText(x, f"device {i}: {verdict}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if verdict == "RE-AIM" else (0, 180, 0), 2)
    tiles.append(x)
while len(tiles) % 2 or len(tiles) < 4:
    tiles.append(np.zeros((360, 640, 3), np.uint8))
cv2.imwrite(str(Y / "captures/aim_check.jpg"), np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]))
