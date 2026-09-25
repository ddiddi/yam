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
    """The bottle axis (x, y) near `guess`, or None when fewer than 12 tape-bordered edge points or a poor fit."""
    guess = np.asarray(guess, dtype=float)[:2]
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
        # keep an edge only if blue tape lies just outside it and the run is not absurdly wide
        for u, out in ((a, a - 4), (b, b + 4)):
            if 0 <= out < W and blue[v, out] and (b - a) < 400:
                pts.append((u, v))
    if len(pts) < 12:
        if not quiet:
            print(f"only {len(pts)} tape-bordered edge points", file=sys.stderr)
        return None
    rays = [cam.ray(float(u), float(v)) for u, v in pts]

    def resid(p):
        out = []
        for o, dv in rays:
            zs = np.linspace(0, 0.10, 21)
            P = o[None] + ((zs - o[2]) / dv[2])[:, None] * dv[None]
            out.append(np.min(np.hypot(P[:, 0] - p[0], P[:, 1] - p[1])) - diameter / 2)
        return np.array(out)

    s = least_squares(resid, guess, loss="soft_l1", f_scale=0.002)
    rms = float(np.sqrt(np.mean(s.fun ** 2)))
    if rms > 0.004:
        if not quiet:
            print(f"bad fit: rms {rms * 1000:.1f} mm", file=sys.stderr)
        return None
    if not quiet:
        print(f"body fit: {len(pts)} tape-bordered edge points, rms {rms * 1000:.1f} mm", file=sys.stderr)
    return float(s.x[0]), float(s.x[1])


if __name__ == "__main__":
    from calib3d import load_cameras
    from scene3d import grab

    g = [float(v) for v in sys.argv[1].split(",")]
    d = float(sys.argv[2]) if len(sys.argv) > 2 else D_DEFAULT
    r = find(grab(1), load_cameras()["cam1"], g, d)
    if r is not None:
        print(f"{r[0]:.4f},{r[1]:.4f},{d:.4f}")
