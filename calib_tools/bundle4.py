"""All cameras at once: lens (f, k1) and pose per camera, plus the scale-top label corners (flat at the scale top), fitted
together to (a) the fingertip sweeps (robot-frame FK) and (b) the label corners each camera sees now (calib_tools/
plate_labels.py). The labels tie the cameras to each other; the fingertips tie them all to the arm. Robust (soft-L1)
residuals, outliers re-selected between rounds.

    YAM_CAM_MAP=... python calib_tools/bundle4.py sweep_zone15,sweep_zone16 [--top-z 0.052] [--write]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

Y = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Y)); sys.path.insert(0, str(Y / "calib_tools"))
from plate_labels import label_corners  # noqa: E402
from scene3d import grab  # noqa: E402

dirs = sys.argv[1].split(",")
top = float(sys.argv[sys.argv.index("--top-z") + 1]) if "--top-z" in sys.argv else 0.052
write = "--write" in sys.argv
KEYS = ["cam0", "cam1", "cam2", "cam3"]
W, H = 1280, 720
recs = [r for d in dirs for r in json.loads((Y / "captures" / d / "sweep.json").read_text())]
# label corners seen now (a corner on the frame edge is cut off: dropped)
lab = {}
for k in KEYS:
    f = grab(int(k[3:]))
    cv2.imwrite(str(Y / f"captures/labels_{k}.png"), f)
    c = label_corners(f)
    if c is None:
        continue
    ok = (c[:, 0] > 4) & (c[:, 0] < W - 5) & (c[:, 1] > 4) & (c[:, 1] < H - 5)
    lab[k] = {i: c[i] for i in range(8) if ok[i]}
print("label corners seen:", {k: sorted(v) for k, v in lab.items()})
# start: per camera, RANSAC PnP over a focal-length scan on its fingertips
init = {}
for k in KEYS:
    P = np.array([r["fk"] for r in recs if r["px"].get(k)], float); I = np.array([r["px"][k] for r in recs if r["px"].get(k)], float)
    best = None
    for f0 in range(900, 2301, 50):
        K = np.array([[f0, 0, W / 2], [0, f0, H / 2], [0, 0, 1]])
        ok, rv, tv, inl = cv2.solvePnPRansac(P, I, K, np.zeros(5), reprojectionError=10.0, iterationsCount=3000, confidence=0.999)
        if not ok or inl is None or len(inl) < 6:
            continue
        C = (-cv2.Rodrigues(rv)[0].T @ tv).ravel()
        if C[2] < 0.1:
            continue
        e = np.linalg.norm(cv2.projectPoints(P, rv, tv, K, np.zeros(5))[0].reshape(-1, 2) - I, axis=1)
        sc = ((e < 10).sum(), -np.median(e[e < 10]))
        if best is None or sc > best[0]:
            best = (sc, f0, rv.ravel(), tv.ravel())
    init[k] = best
    print(f"{k}: start f={best[1]} with {best[0][0]} fingertips")
# label corners in the world: start from cam1's view (it is tied to the arm by the most fingertips)
ref = max(lab, key=lambda k: init[k][0][0] if init.get(k) else -1)
f0, rv0, tv0 = init[ref][1], init[ref][2], init[ref][3]
K0 = np.array([[f0, 0, W / 2], [0, f0, H / 2], [0, 0, 1]])
R0 = cv2.Rodrigues(rv0)[0]; C0 = -R0.T @ tv0
L0 = {}
for i in range(8):
    obs = [(k, lab[k][i]) for k in lab if i in lab[k]]
    k, uv = next(((k, uv) for k, uv in obs if k == ref), obs[0]) if obs else (None, None)
    if k is None:
        continue
    fk, rvk, tvk = init[k][1], init[k][2], init[k][3]
    Kk = np.array([[fk, 0, W / 2], [0, fk, H / 2], [0, 0, 1]]); Rk = cv2.Rodrigues(rvk)[0]; Ck = -Rk.T @ tvk
    d = Rk.T @ np.array([(uv[0] - W / 2) / fk, (uv[1] - H / 2) / fk, 1.0])
    L0[i] = (Ck + (top - Ck[2]) / d[2] * d)[:2]
lids = sorted(L0)
print("label corners used:", lids)


def unpack(p):
    cams = {}
    for j, k in enumerate(KEYS):
        q = p[8 * j: 8 * j + 8]
        cams[k] = (np.array([[q[0], 0, W / 2], [0, q[0], H / 2], [0, 0, 1]]), np.array([q[1], 0, 0, 0, 0.0]), q[2:5], q[5:8])
    L = p[32:].reshape(-1, 2)
    return cams, {i: np.r_[L[n], top] for n, i in enumerate(lids)}


p0 = np.concatenate([np.r_[init[k][1], 0.0, init[k][2], init[k][3]] for k in KEYS] + [np.array([L0[i] for i in lids]).ravel()])
tips = {k: (np.array([r["fk"] for r in recs if r["px"].get(k)], float), np.array([r["px"][k] for r in recs if r["px"].get(k)], float)) for k in KEYS}
use = {}
for k in KEYS:
    K, d, rv, tv = unpack(p0)[0][k]
    e = np.linalg.norm(cv2.projectPoints(tips[k][0], rv, tv, K, d)[0].reshape(-1, 2) - tips[k][1], axis=1)
    use[k] = e < 15


def resid(p):
    cams, L = unpack(p)
    out = []
    for k in KEYS:
        K, d, rv, tv = cams[k]
        P, I = tips[k]
        if use[k].any():
            out.append((cv2.projectPoints(P[use[k]], rv, tv, K, d)[0].reshape(-1, 2) - I[use[k]]).ravel())
        if k in lab:
            ids = [i for i in lids if i in lab[k]]
            if ids:
                pr = cv2.projectPoints(np.array([L[i] for i in ids]), rv, tv, K, d)[0].reshape(-1, 2)
                out.append(2.0 * (pr - np.array([lab[k][i] for i in ids])).ravel())  # labels: sharp corners, weight x2
        out.append([0.02 * (p[8 * KEYS.index(k) + 1]) * 100])  # keep k1 small unless the data insist
    return np.concatenate(out)


p = p0
for rnd in range(3):
    s = least_squares(resid, p, loss="soft_l1", f_scale=3.0, x_scale="jac", max_nfev=3000)
    p = s.x
    cams, L = unpack(p)
    for k in KEYS:  # re-select fingertip inliers with the joint model
        K, d, rv, tv = cams[k]
        e = np.linalg.norm(cv2.projectPoints(tips[k][0], rv, tv, K, d)[0].reshape(-1, 2) - tips[k][1], axis=1)
        use[k] = e < (12 if rnd < 2 else 8)
cams, L = unpack(p)
out = json.loads((Y / "cameras.json").read_text())
for k in KEYS:
    K, d, rv, tv = cams[k]
    R = cv2.Rodrigues(np.asarray(rv))[0]; C = -R.T @ np.asarray(tv)
    e = np.linalg.norm(cv2.projectPoints(tips[k][0], rv, tv, K, d)[0].reshape(-1, 2) - tips[k][1], axis=1)
    el = [np.linalg.norm(cv2.projectPoints(np.array([L[i]]), rv, tv, K, d)[0].ravel() - lab[k][i]) for i in lids if k in lab and i in lab[k]]
    print(f"{k}: f={K[0, 0]:.0f}px k1={d[0]:+.3f}; fingertips {use[k].sum()}/{len(e)} within 8 px (median {np.median(e[use[k]]) if use[k].any() else float('nan'):.1f}); "
          f"labels {len(el)} (median {np.median(el) if el else float('nan'):.1f} px); camera at {np.round(C, 3)}")
    out[k] = {"K": K.tolist(), "dist": d.tolist(), "R": R.tolist(), "t": np.ravel(tv).tolist(), "size": [W, H],
              "rms_px": float(np.sqrt(np.mean(e[use[k]] ** 2))) if use[k].any() else None, "center": C.tolist(),
              "n_points": int(use[k].sum() + len(el)), "fov_deg": float(2 * np.degrees(np.arctan(W / 2 / K[0, 0]))),
              "method": f"bundle: fingertip sweeps ({', '.join(dirs)}) + scale-top label corners shared by all views"}
print("label corners (world):", {i: np.round(L[i][:2], 4).tolist() for i in lids})
if write:
    (Y / "cameras.json").write_text(json.dumps(out, indent=1))
    (Y / "captures/scale_labels_world.json").write_text(json.dumps({str(i): L[i].tolist() for i in lids}, indent=1))
    print("wrote cameras.json and captures/scale_labels_world.json")
