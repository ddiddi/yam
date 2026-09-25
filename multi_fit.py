"""Locate an upright cylinder (bottle, dish) from EVERY calibrated camera at once: in each view the object's outline
(its difference from that camera's empty-zone background, near the guess) gives left/right edge pixels per row; each
edge is a ray tangent to the cylinder. One least-squares fit of x, y (radius and height known) over all rays; per-camera
residuals show which views agree.

    YAM_CAM_MAP=... python multi_fit.py x,y radius height [--show out.jpg]   -> "x,y" and a per-camera report
    from multi_fit import fit; fit((x, y), r, h) -> ((x, y), report) or (None, report)
"""
from __future__ import annotations

import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

from calib3d import load_cameras
from scene3d import BACKGROUND, grab


def _edges(cam, frame, bg, guess, r, h, reach=0.03):
    th = np.linspace(0, 2 * np.pi, 48)
    ring = [np.c_[guess[0] + (r + reach) * np.cos(th), guess[1] + (r + reach) * np.sin(th), np.full(48, z)] for z in (0.0, h)]
    uv = np.vstack([cam.project(p) for p in ring]).astype(np.int32)
    Hh, Ww = frame.shape[:2]
    roi = np.zeros((Hh, Ww), np.uint8)
    cv2.fillConvexPoly(roi, cv2.convexHull(uv), 1)
    a = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.int16)
    b = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.int16)
    diff = np.abs(a - b).sum(2)
    m = ((diff > 28) & (roi > 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return [], m
    # the blob that covers the guess's own footprint most (not simply the biggest: a bottle beside a dish is bigger)
    th2 = np.linspace(0, 2 * np.pi, 36)
    foot = np.zeros((Hh, Ww), np.uint8)
    for z in (0.0, h):
        cv2.fillConvexPoly(foot, cam.project(np.c_[guess[0] + r * np.cos(th2), guess[1] + r * np.sin(th2),
                                                   np.full(36, z)]).astype(np.int32), 1)
    score = [((lab == i) & (foot > 0)).sum() if st[i, cv2.CC_STAT_AREA] >= 400 else -1 for i in range(1, n)]
    if max(score) <= 0:
        return [], m
    k = 1 + int(np.argmax(score))
    blob = lab == k
    ys = np.nonzero(blob.any(1))[0]
    pts = []
    for v in ys[::3]:
        xs = np.nonzero(blob[v])[0]
        if len(xs) < 6 or xs[0] <= 1 or xs[-1] >= Ww - 2:  # a row cut by the frame edge has no true outline
            continue
        pts += [(float(xs[0]), float(v)), (float(xs[-1]), float(v))]
    return pts, blob.astype(np.uint8)


def fit(guess, r, h, keys=None, show=None):
    cams = load_cameras()
    keys = keys or sorted(cams)
    rays, owner, report, views = [], [], {}, {}
    for k in keys:
        f = grab(int(k[3:]))
        bg = cv2.imread(str(BACKGROUND[k]))
        if bg is None:
            continue
        pts, blob = _edges(cams[k], f, bg, np.asarray(guess, float), r, h)
        views[k] = (f, pts)
        for u, v in pts:
            rays.append(cams[k].ray(u, v))
            owner.append(k)
    if len(rays) < 10:
        return None, {"error": f"only {len(rays)} outline points"}
    owner = np.array(owner)

    def resid(p):
        out = []
        for o, d in rays:
            zs = np.linspace(0.0, h, 9)
            P = o[None] + ((zs - o[2]) / d[2])[:, None] * d[None]
            out.append(np.min(np.hypot(P[:, 0] - p[0], P[:, 1] - p[1])) - r)
        return np.array(out)

    s = least_squares(resid, np.asarray(guess, float), loss="soft_l1", f_scale=0.004)
    e = np.abs(resid(s.x))
    for k in keys:
        sel = owner == k
        if sel.any():
            report[k] = {"points": int(sel.sum()), "median_mm": round(float(np.median(e[sel])) * 1000, 1)}
    if show:
        th = np.linspace(0, 2 * np.pi, 60)
        tiles = []
        for k, (f, pts) in views.items():
            g = f.copy()
            for u, v in pts:
                cv2.circle(g, (int(u), int(v)), 2, (0, 255, 0), -1)
            for z in (0.0, h):
                Q = np.c_[s.x[0] + r * np.cos(th), s.x[1] + r * np.sin(th), np.full(60, z)]
                cv2.polylines(g, [cams[k].project(Q).astype(np.int32)], True, (0, 0, 255), 2)
            cv2.putText(g, k, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 3)
            tiles.append(cv2.resize(g, (640, 360)))
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(show, np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))
    ok = np.median(e) < 0.006
    return ((float(s.x[0]), float(s.x[1])) if ok else None), report


if __name__ == "__main__":
    g = [float(v) for v in sys.argv[1].split(",")]
    r, h = float(sys.argv[2]), float(sys.argv[3])
    show = sys.argv[sys.argv.index("--show") + 1] if "--show" in sys.argv else None
    xy, rep = fit(g, r, h, show=show)
    print(rep, file=sys.stderr)
    if xy:
        print(f"{xy[0]:.4f},{xy[1]:.4f}")
