"""Robust per-camera calibration from fingertip sweeps when the automatic detections are noisy: RANSAC PnP with the
lens fixed at a prior (tried over a range of focal lengths), then the focal length, k1 and pose refined on the
inliers only. Principal point at the image centre.

    python calib_tools/solve_ransac.py sweep_zone11,sweep_zone12 [--write]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

Y = Path(__file__).resolve().parents[1]
dirs = sys.argv[1].split(",")
write = "--write" in sys.argv
recs = [r for d in dirs for r in json.loads((Y / "captures" / d / "sweep.json").read_text())]
keys = sorted({k for r in recs for k in r["px"]})
W, H = 1280, 720
out = json.loads((Y / "cameras.json").read_text())
for k in keys:
    P = np.array([r["fk"] for r in recs if r["px"].get(k) is not None], np.float64)
    I = np.array([r["px"][k] for r in recs if r["px"].get(k) is not None], np.float64)
    best = None
    for f in np.arange(800, 2001, 50):
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
        d = np.zeros(5)
        ok, rv, tv, inl = cv2.solvePnPRansac(P, I, K, d, reprojectionError=10.0, iterationsCount=4000, confidence=0.999)
        if not ok or inl is None or len(inl) < 6:
            continue
        inl = inl.ravel()
        rv, tv = cv2.solvePnPRefineLM(P[inl], I[inl], K, d, rv, tv)
        e = np.linalg.norm(cv2.projectPoints(P[inl], rv, tv, K, d)[0].reshape(-1, 2) - I[inl], axis=1)
        C = (-cv2.Rodrigues(rv)[0].T @ tv).ravel()
        if C[2] < 0.1 or np.linalg.norm(C[:2]) > 2.0:  # a camera below the table or across the room: wrong
            continue
        score = (len(inl), -np.sqrt(np.mean(e ** 2)))
        if best is None or score > best[0]:
            best = (score, f, rv, tv, inl)
    if best is None:
        print(f"{k}: no consistent pose from {len(P)} detections")
        continue
    (n, neg_rms), f, rv, tv, inl = best
    # refine f and k1 on the inliers (non-planar points: calibrateCamera may move f)
    K0 = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
             | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    K, d = K0, np.zeros(5)
    if len(inl) >= 10:
        rms, K1, d1, rvs, tvs = cv2.calibrateCamera([P[inl].astype(np.float32)], [I[inl].astype(np.float32)], (W, H), K0.copy(),
                                                     None, flags=flags)
        if 0.6 * f < K1[0, 0] < 1.6 * f and abs(d1.ravel()[0]) < 0.5:
            K, d, rv, tv = K1, d1.ravel(), rvs[0], tvs[0]
    e_all = np.linalg.norm(cv2.projectPoints(P, rv, tv, K, d)[0].reshape(-1, 2) - I, axis=1)
    R = cv2.Rodrigues(rv)[0]
    C = (-R.T @ tv).ravel()
    ein = e_all[inl]
    print(f"{k}: {len(inl)}/{len(P)} inliers, rms {np.sqrt(np.mean(ein ** 2)):.1f} px, f={K[0, 0]:.0f}px "
          f"({2 * np.degrees(np.arctan(W / 2 / K[0, 0])):.0f} deg HFOV), k1={d[0]:+.3f}, camera at {np.round(C, 3)}")
    if write:
        out[k] = {"K": K.tolist(), "dist": list(d), "R": R.tolist(), "t": np.ravel(tv).tolist(), "size": [W, H],
                  "rms_px": float(np.sqrt(np.mean(ein ** 2))), "center": C.tolist(), "n_points": int(len(inl)),
                  "n_detected": int(len(P)), "fov_deg": float(2 * np.degrees(np.arctan(W / 2 / K[0, 0]))),
                  "method": f"RANSAC PnP over a focal-length scan + f/k1 refine on inliers ({', '.join(dirs)})"}
if write:
    (Y / "cameras.json").write_text(json.dumps(out, indent=1))
    print("wrote cameras.json")
