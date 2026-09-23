"""Petri dish centre from cam1 by its beige agar colour (shadows on the white sheet fool the background diff).
Prints "x,y" or nothing. --show writes an overlay."""
import sys
import cv2
import numpy as np
sys.path.insert(0, "/Users/solotech/yam")
import pick_place as pp
from scene3d import grab

f = grab(1)
m = pp.load_cameras()[pp.DET]
Z = pp.load_zone()
zm = np.zeros(f.shape[:2], np.uint8)
cv2.fillPoly(zm, [m.project(np.c_[Z, np.zeros(len(Z))]).astype(np.int32)], 255)
hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
mask = ((hsv[..., 0] >= 12) & (hsv[..., 0] <= 40) & (hsv[..., 1] >= 35) & (hsv[..., 2] >= 90) & (zm > 0)).astype(np.uint8) * 255
mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
if not cnts or cv2.contourArea(max(cnts, key=cv2.contourArea)) < 3000:
    sys.exit(0)
c = max(cnts, key=cv2.contourArea)
(u, v), _, _ = cv2.fitEllipse(c)
xy = m.hit_plane(u, v, 0.010)[:2]  # the agar surface sits ~1 cm up
print(f"{xy[0]:.3f},{xy[1]:.3f}")
if "--show" in sys.argv:
    g = f.copy()
    for z in (0.0, 0.015):
        ring = np.array([[xy[0] + 0.045 * np.cos(t), xy[1] + 0.045 * np.sin(t), z] for t in np.linspace(0, 2 * np.pi, 60)])
        cv2.polylines(g, [m.project(ring).astype(np.int32)], True, (0, 0, 255), 1)
    cv2.drawContours(g, [c], -1, (0, 255, 0), 1)
    cv2.imwrite(sys.argv[sys.argv.index("--show") + 1], cv2.resize(g[330:650, 450:900], (675, 480)))
