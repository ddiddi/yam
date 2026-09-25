"""Petri dish centre on the blue tape from cam1: the dish is the non-blue patch near where it was last seen; a
9 cm x 1.5 cm cylinder is fitted to its outline (rays graze the rim between the table and its top).

    python scripts/dish_find_rim.py 0.306,0.112 [--show out.jpg]    # -> "x,y" or nothing (not found / bad fit)
"""
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, "/Users/solotech/yam")
from calib3d import load_cameras  # noqa: E402
from scene3d import grab  # noqa: E402

R, H = 0.045, 0.015
near = np.array([float(v) for v in sys.argv[1].split(",")])
cam = load_cameras()["cam1"]
f = grab(1)
gu, gv = cam.project(np.array([[near[0], near[1], H / 2]]))[0].astype(int)
hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
notblue = ~((hsv[..., 0] > 95) & (hsv[..., 0] < 125) & (hsv[..., 1] > 90))
win = np.zeros(f.shape[:2], bool)
win[max(0, gv - 110):gv + 110, max(0, gu - 150):gu + 150] = True
m = (notblue & win).astype(np.uint8) * 255
m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
n, lab, st, cen = cv2.connectedComponentsWithStats(m)
if n < 2:
    sys.exit(0)
j = 1 + int(np.argmin([np.hypot(*(cen[i] - (gu, gv))) if st[i, 4] > 500 else 1e9 for i in range(1, n)]))
if st[j, 4] <= 500:
    sys.exit(0)
pts = cv2.findContours((lab == j).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0].reshape(-1, 2).astype(float)
rays = [cam.ray(u, v) for u, v in pts[::2]]


def resid(p, r):
    out = []
    for o, dv in rays:
        zs = np.linspace(0, H, 7)
        P = o[None] + ((zs - o[2]) / dv[2])[:, None] * dv[None]
        out.append(np.min(np.abs(np.hypot(P[:, 0] - p[0], P[:, 1] - p[1]) - r)))
    return np.array(out)


x0 = np.mean([cam.hit_plane(u, v, H / 2)[:2] for u, v in pts], axis=0)
fixed = least_squares(lambda p: resid(p, R), x0, loss="soft_l1", f_scale=0.003)
free = least_squares(lambda p: resid(p[:2], p[2]), np.r_[x0, R], loss="soft_l1", f_scale=0.003)
rms = float(np.sqrt(np.mean(fixed.fun ** 2)))
# a real dish: its own radius comes out near 4.5 cm and the fixed-radius fit hugs the outline
if not (0.040 <= free.x[2] <= 0.050) or rms > 0.003 or np.linalg.norm(fixed.x - near) > 0.05:
    print(f"bad fit: r {free.x[2] * 100:.1f} cm, rms {rms * 1000:.1f} mm, {np.linalg.norm(fixed.x - near) * 100:.1f} cm from "
          "the last position", file=sys.stderr)
    sys.exit(0)
print(f"{fixed.x[0]:.4f},{fixed.x[1]:.4f}")
if "--show" in sys.argv:
    g = f.copy()
    cv2.drawContours(g, [pts.astype(np.int32).reshape(-1, 1, 2)], -1, (0, 0, 255), 1)
    for z in (0.0, H):
        ring = np.array([[fixed.x[0] + R * np.cos(t), fixed.x[1] + R * np.sin(t), z] for t in np.linspace(0, 2 * np.pi, 60)])
        cv2.polylines(g, [cam.project(ring).astype(np.int32)], True, (0, 255, 0), 1)
    cv2.imwrite(sys.argv[sys.argv.index("--show") + 1], g)
