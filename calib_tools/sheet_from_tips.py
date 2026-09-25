"""The camera is fixed to the ChArUco sheet by PnP on its corners (f scanned on held-out corners); the sheet's own pose
on the scale top (x, y, yaw at --top-z) is the only unknown, fitted to the fingertip sweep (robot-frame FK) with RANSAC.
Gives the camera in the robot frame and where the scale now stands.

    python calib_tools/sheet_from_tips.py SWEEP_KEY LIVE_INDEX sweep_zone13,sweep_zone14 [--top-z 0.052] [--write]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

Y = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Y)); sys.path.insert(0, str(Y / "calib_tools"))
import station_calib as sc  # noqa: E402
from scene3d import grab  # noqa: E402

key, live, dirs = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
top = float(sys.argv[sys.argv.index("--top-z") + 1]) if "--top-z" in sys.argv else 0.052
write = "--write" in sys.argv
st = json.loads((Y / "station.json").read_text())
b, mdet, cdet = sc.board(0.75)
ids_all = list(b.getIds().ravel()); objp = b.getObjPoints()
f = grab(live)
g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
cc, ci, mc, mi = cdet.detectBoard(g)
Bp = [np.asarray(b.getChessboardCorners())[ci.ravel()]]; Bi = [cc.reshape(-1, 2)]
for i, m in zip(mi.ravel(), mc):
    if int(i) in ids_all:
        Bp.append(objp[ids_all.index(int(i))]); Bi.append(m.reshape(4, 2))
Bxy = sc.mirror(np.vstack(Bp)); Bi = np.vstack(Bi).astype(float)
B3 = np.c_[Bxy, np.zeros(len(Bxy))]  # sheet frame: flat, z up
# 1. camera relative to the sheet, lens scanned on held-out corners
hold = np.zeros(len(B3), bool); hold[::3] = True
best = None
for fx in range(1100, 2001, 10):
    K = np.array([[fx, 0, 640], [0, fx, 360], [0, 0, 1]], float)
    ok, rv, tv, _ = cv2.solvePnPRansac(B3[~hold], Bi[~hold], K, np.zeros(5), reprojectionError=3.0, iterationsCount=2000)
    if not ok:
        continue
    rv, tv = cv2.solvePnPRefineLM(B3[~hold], Bi[~hold], K, np.zeros(5), rv, tv)
    e = np.linalg.norm(cv2.projectPoints(B3, rv, tv, K, np.zeros(5))[0].reshape(-1, 2) - Bi, axis=1)
    if best is None or np.median(e[hold]) < best[0]:
        best = (np.median(e[hold]), fx, rv, tv, np.sqrt(np.mean(e[~hold] ** 2)))
hmed, fx, rv, tv, fit = best
K = np.array([[fx, 0, 640], [0, fx, 360], [0, 0, 1]], float)
Rcs = cv2.Rodrigues(rv)[0]  # sheet -> camera
print(f"{len(B3)} sheet points: f={fx}px, fit {fit:.2f} px, held-out median {hmed:.2f} px")
# 2. the sheet in the robot frame: x, y, yaw (flat at top); world -> sheet: p_s = Rz(-yaw) (p_w - [x, y, top])
recs = [r for d in dirs for r in json.loads((Y / "captures" / d / "sweep.json").read_text()) if r["px"].get(key)]
Tp = np.array([r["fk"] for r in recs]); Ti = np.array([r["px"][key] for r in recs], float)


def proj(q, P):
    c, s = np.cos(q[2]), np.sin(q[2])
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    Ps = (P - np.array([q[0], q[1], top])) @ Rz  # rows: Rz^T (p - o)
    return cv2.projectPoints(Ps, rv, tv, K, np.zeros(5))[0].reshape(-1, 2)


A = sc.mirror(np.array([[0, 0, 0], [0.165, 0, 0], [0.165, 0.12, 0], [0, 0.12, 0]], float))
R2, t2 = sc.rigid2d(A, np.array(st["board_outline"]))
q0 = np.array([t2[0], t2[1], np.arctan2(R2[1, 0], R2[0, 0])])
rng = np.random.default_rng(0)
bestq = None
for _ in range(400):  # RANSAC over triples of fingertips
    idx = rng.choice(len(Tp), 3, replace=False)
    s_ = least_squares(lambda q: (proj(q, Tp[idx]) - Ti[idx]).ravel(), q0 + rng.normal(0, [0.05, 0.05, 0.1]))
    e = np.linalg.norm(proj(s_.x, Tp) - Ti, axis=1)
    n = (e < 12).sum()
    if bestq is None or n > bestq[0]:
        bestq = (n, s_.x)
inl = np.linalg.norm(proj(bestq[1], Tp) - Ti, axis=1) < 12
s_ = least_squares(lambda q: (proj(q, Tp[inl]) - Ti[inl]).ravel(), bestq[1])
q = s_.x
e = np.linalg.norm(proj(q, Tp) - Ti, axis=1)
c, s = np.cos(q[2]), np.sin(q[2])
Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]); o = np.array([q[0], q[1], top])
R = Rcs @ Rz.T; t = tv.ravel() - R @ o  # world -> camera
C = -R.T @ t
print(f"fingertips: {inl.sum()}/{len(Tp)} agree, median {np.median(e[inl]):.1f} px; camera at {np.round(C, 3)}")
print(f"sheet origin moved {np.round(q[:2] - t2, 4)} m ({np.linalg.norm(q[:2] - t2) * 100:.1f} cm), turned {np.degrees(q[2] - q0[2]):+.2f} deg")
if write:
    cams = json.loads((Y / "cameras.json").read_text())
    cams[f"cam{live}"] = {"K": K.tolist(), "dist": [0, 0, 0, 0, 0], "R": R.tolist(), "t": t.tolist(), "size": [1280, 720],
                          "rms_px": float(fit), "center": C.tolist(), "n_points": int(len(B3) + inl.sum()),
                          "fov_deg": float(2 * np.degrees(np.arctan(640 / fx))),
                          "method": f"PnP on the ChArUco sheet + the sheet's pose fitted to {inl.sum()} fingertips ({', '.join(dirs)}, {key})"}
    (Y / "cameras.json").write_text(json.dumps(cams, indent=1))
    Rm = np.array([[np.cos(q[2] - q0[2]), -np.sin(q[2] - q0[2])], [np.sin(q[2] - q0[2]), np.cos(q[2] - q0[2])]])
    mv = lambda p_: ((np.asarray(p_) - t2) @ Rm.T + q[:2]).round(4).tolist()
    (Y / "captures/station_before_4cam.json").write_text(json.dumps(st, indent=1))
    st.update(board_outline=mv(st["board_outline"]), board_centre=mv(st["board_centre"]), footprint=mv(st["footprint"]),
              place=mv(st["place"]), board_yaw_deg=float(np.degrees(q[2])), placed_by=f"cam{live}: sheet PnP + fingertips")
    (Y / "station.json").write_text(json.dumps(st, indent=1))
    print(f"wrote cameras.json[cam{live}] and station.json (old one in captures/station_before_4cam.json)")
