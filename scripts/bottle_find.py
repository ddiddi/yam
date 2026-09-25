"""Upright bottle on the blue tape from cam1: a vertical cylinder of known diameter fitted to the edges of its lower
body where they border the blue tape (the desk, cables or the white sheet beside it are not trusted).

    python scripts/bottle_find.py x,y [diameter]  ->  "x,y,diameter" or nothing
    from bottle_find import find; find(frame, cam, (x, y))  -> (x, y) or None   (pick_place.py --refind uses this)
"""
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, "/Users/solotech/yam")

D_DEFAULT = 0.0785


def find(frame: np.ndarray, cam, guess, diameter: float = D_DEFAULT, quiet: bool = False):
    """The bottle axis (x, y) near `guess`. The rows the edges are read from follow the guess, so a guess a few cm
    off reads the wrong part of the body: a small grid of starts around it is tried, the best fit kept."""
    guess = np.asarray(guess, dtype=float)[:2]
    best = None
    for dx in (0.0, 0.02, -0.02, 0.04, -0.04):
        for dy in (0.0, 0.015, -0.015, 0.03, -0.03):
            r = _fit(frame, cam, guess + (dx, dy), diameter)
            if r is not None and (best is None or r[1] < best[1]):
                best = r
        if best is not None and best[1] < 0.002:
            break
    if best is None or np.linalg.norm(np.array(best[0]) - guess) > 0.06:
        if not quiet:
            print("no consistent body fit near the guess", file=sys.stderr)
        return None
    if not quiet:
        print(f"body fit: rms {best[1] * 1000:.1f} mm", file=sys.stderr)
    return best[0]


def _fit(frame: np.ndarray, cam, guess, diameter: float):
    """One fit from one start: ((x, y), rms) or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = (hsv[..., 0] > 95) & (hsv[..., 0] < 125) & (hsv[..., 1] > 90)
    pale = (hsv[..., 1] < 70) & (hsv[..., 2] > 110)
    ub, vb = cam.project(np.array([[*guess, 0.0]]))[0]
    ut, vt = cam.project(np.array([[*guess, 0.08]]))[0]
    W = frame.shape[1]
    pts = []
    for v in range(int(vt), int(vb) + 15, 2):
        if not 0 <= v < frame.shape[0]:
            continue
        uc = int(ub + (ut - ub) * (v - vb) / (vt - vb))
        # the axis column may sit on a label or a reflection: take the nearest pale pixel within 25 px
        cands = [u for u in range(uc - 25, uc + 26) if 0 <= u < W and pale[v, u]]
        if not cands:
            continue
        uc = min(cands, key=lambda u: abs(u - uc))
        a = b = uc
        while a > 0 and not blue[v, a - 1]:
            a -= 1
        while b < W - 1 and not blue[v, b + 1]:
            b += 1
        # keep an edge only if blue tape lies just outside it (the desk reads as pale as the bottle in some light,
        # and the white sheet always does) and the run is not absurdly wide
        for u, out in ((a, a - 4), (b, b + 4)):
            if 0 <= out < W and blue[v, out] and (b - a) < 400:
                pts.append((u, v))
    # the front of the base rim: the lowest bottle pixel under the axis, against the tape it stands on. Its ray
    # meets the table one radius in front of the axis - this pins the depth when only one side has tape edges
    base = None
    col = int(ub)
    if 0 <= col < W:
        v = int(vb) - 25
        while v < frame.shape[0] - 1 and not blue[v + 1, col]:
            v += 1
        if v < frame.shape[0] - 1 and v > int(vb) - 25:
            P = cam.hit_plane(float(col), float(v), 0.0)[:2]
            d = P - np.asarray(cam.C[:2], dtype=float)
            base = P + (diameter / 2) * d / (np.linalg.norm(d) + 1e-9)
    if len(pts) < 12:
        return None
    rays = [cam.ray(float(u), float(v)) for u, v in pts]

    def resid(p):
        out = []
        for o, dv in rays:
            zs = np.linspace(0, 0.13, 27)  # the straight body, base to shoulder
            P = o[None] + ((zs - o[2]) / dv[2])[:, None] * dv[None]
            out.append(np.min(np.hypot(P[:, 0] - p[0], P[:, 1] - p[1])) - diameter / 2)
        if base is not None:  # worth ~5 edge points
            out += list(np.sqrt(5.0) * (np.asarray(p) - base))
        return np.array(out)

    s = least_squares(resid, guess, loss="soft_l1", f_scale=0.002)
    rms = float(np.sqrt(np.mean(s.fun ** 2)))
    if rms > 0.004:
        return None
    return (float(s.x[0]), float(s.x[1])), rms


if __name__ == "__main__":
    from calib3d import load_cameras
    from scene3d import grab

    g = [float(v) for v in sys.argv[1].split(",")]
    d = float(sys.argv[2]) if len(sys.argv) > 2 else D_DEFAULT
    r = find(grab(1), load_cameras()["cam1"], g, d)
    if r is not None:
        print(f"{r[0]:.4f},{r[1]:.4f},{d:.4f}")
