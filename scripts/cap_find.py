"""A wash bottle's orange screw cap from cam1: vertical-cylinder fit to the cap's left/right edges -> the bottle
axis xy and cap diameter; then the cap's bottom/top heights where the rays of its lowest/highest rows meet that
axis. python scripts/cap_find.py [--show out.jpg] -> "x,y,diameter,z_bottom,z_top" or nothing."""
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, "/Users/solotech/yam")
from calib3d import load_cameras  # noqa: E402
from scene3d import grab  # noqa: E402

c = load_cameras()["cam1"]
arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
f = cv2.imread(arg("--frame")) if arg("--frame") else grab(1)
AXIS = np.array([float(v) for v in arg("--axis").split(",")]) if arg("--axis") else None
ZS = [float(v) for v in arg("--z").split(",")] if arg("--z") else None
hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
orange = ((hsv[..., 0] >= 5) & (hsv[..., 0] <= 22) & (hsv[..., 1] > 150) & (hsv[..., 2] > 150)).astype(np.uint8)
orange = cv2.morphologyEx(cv2.morphologyEx(orange, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
n, lab, st, cen = cv2.connectedComponentsWithStats(orange)
if n < 2:
    sys.exit(0)
# the cap: of the sizeable orange blobs, the one widest across its bottom rows (the spout tube is long but thin)
def bottom_width(i):
    mm = lab == i
    yy = np.where(mm.any(1))[0]
    return float(np.median([np.ptp(np.where(mm[y])[0]) + 1 for y in yy[-15:]])) if st[i, 4] > 800 else 0.0


j = max(range(1, n), key=bottom_width)
m = lab == j
ys = np.where(m.any(1))[0]
# the cap is the wide bottom of the orange blob (the spout tube rises out of it, thin): walk up from the lowest
# row while the row is still cap-wide
w = {y: np.ptp(np.where(m[y])[0]) + 1 for y in ys}
base_w = np.median([w[y] for y in ys[-15:]])
cap_rows = []
for y in ys[::-1]:
    if w[y] < 0.6 * base_w:
        if cap_rows:
            break
        continue  # the rounded bottom edge: not wide yet
    cap_rows.append(y)
full = np.array([y for y in cap_rows if w[y] > 0.8 * max(w[q] for q in cap_rows)])
rows = [(y, np.where(m[y])[0].min(), np.where(m[y])[0].max()) for y in full]
rays = [c.ray(float(u), float(y)) for y, a, b in rows for u in (a, b)]


def resid(p):
    out = []
    for o, dv in rays:
        zs = np.linspace(ZS[0], ZS[1], 9) if ZS else np.linspace(0.10, 0.22, 25)
        P = o[None] + ((zs - o[2]) / dv[2])[:, None] * dv[None]
        out.append(np.min(np.hypot(P[:, 0] - p[0], P[:, 1] - p[1])) - p[2])
    return np.array(out)


x0 = c.hit_plane(float(np.mean([(a + b) / 2 for _, a, b in rows])), float(np.mean(full)), 0.16)[:2]
if AXIS is not None:  # the axis is known (from the body): only the cap's radius is fitted
    s = least_squares(lambda r: resid(np.r_[AXIS, r]), [0.022], loss="soft_l1", f_scale=0.002)
    s.x = np.r_[AXIS, s.x]
else:
    if ZS:
        x0 = c.hit_plane(float(np.mean([(a + b) / 2 for _, a, b in rows])), float(np.mean(full)), float(np.mean(ZS)))[:2]
    s = least_squares(resid, np.r_[x0, 0.022], loss="soft_l1", f_scale=0.002)
ax = s.x[:2]
rms = float(np.sqrt(np.mean(s.fun ** 2)))


def z_on_axis(u, v):
    """Height where the ray through (u, v) passes closest to the vertical axis at ax."""
    o, dv = c.ray(float(u), float(v))
    t = np.dot(ax - o[:2], dv[:2]) / np.dot(dv[:2], dv[:2])
    return float((o + t * dv)[2])


ucol = int(np.median([(a + b) / 2 for _, a, b in rows]))
z_bot = z_on_axis(ucol, max(cap_rows))  # the cap's lowest row: its front-bottom edge
z_top = z_on_axis(ucol, min(cap_rows))  # its highest cap-wide row: the top rim (the tube rises from there)
print(f"{ax[0]:.4f},{ax[1]:.4f},{2 * s.x[2]:.4f},{z_bot:.4f},{z_top:.4f}")
print(f"cap fit rms {rms * 1000:.1f} mm over {len(rows)} rows", file=sys.stderr)
if "--show" in sys.argv:
    g = f.copy()
    for z in (z_bot, z_top):
        ring = np.array([[ax[0] + s.x[2] * np.cos(t), ax[1] + s.x[2] * np.sin(t), z] for t in np.linspace(0, 2 * np.pi, 60)])
        cv2.polylines(g, [c.project(ring).astype(np.int32)], True, (0, 255, 0), 1)
    for z in (0.0, 0.05, 0.10):
        ring = np.array([[ax[0] + 0.039 * np.cos(t), ax[1] + 0.039 * np.sin(t), z] for t in np.linspace(0, 2 * np.pi, 60)])
        cv2.polylines(g, [c.project(ring).astype(np.int32)], True, (255, 0, 255), 1)
    u, v = c.project(np.array([[ax[0], ax[1], 0.1]]))[0].astype(int)
    cv2.imwrite(sys.argv[sys.argv.index("--show") + 1], cv2.resize(g[max(0, v - 260):v + 220, max(0, u - 200):u + 200], None, fx=1.5, fy=1.5))
