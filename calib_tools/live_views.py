"""Live 2x2 preview of the four cameras (YAM_CAM_MAP names), for aiming them by hand: each tile shows the camera name,
the share of the frame that is blue tape (the workspace), and a centre cross. Press q (or Esc) in the window to close.

    YAM_CAM_MAP="0:0,1:1,2:3,3:2" python calib_tools/live_views.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lerobot_record import LiveCamera  # noqa: E402

cams = {}
for i in range(4):
    try:
        cams[f"cam{i}"] = LiveCamera(i)
    except Exception as e:  # noqa: BLE001
        print(f"cam{i}: {e}")
time.sleep(1.0)
cv2.namedWindow("cameras (q to close)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("cameras (q to close)", 1280, 720)
while True:
    tiles = []
    for k in ("cam0", "cam1", "cam2", "cam3"):
        f = cams[k].latest() if k in cams else None
        if f is None:
            t = np.zeros((360, 640, 3), np.uint8)
            cv2.putText(t, f"{k}: no frames", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            tiles.append(t)
            continue
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        tape = ((hsv[..., 0] > 95) & (hsv[..., 0] < 125) & (hsv[..., 1] > 90)).mean()
        t = cv2.resize(f, (640, 360))
        cv2.drawMarker(t, (320, 180), (0, 255, 255), cv2.MARKER_CROSS, 40, 2)
        cv2.putText(t, f"{k}  tape {tape * 100:.0f}%", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        tiles.append(t)
    cv2.imshow("cameras (q to close)", np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    if cv2.waitKey(100) & 0xFF in (ord("q"), 27):
        break
for c in cams.values():
    c.close()
cv2.destroyAllWindows()
