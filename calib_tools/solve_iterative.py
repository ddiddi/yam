"""Tighten a camera's pose from fingertip sweeps whose automatic detections are noisy: start from a rough pose,
re-detect every saved blink (captures/<sweep>/<i>_<cam>_{a,a2,b,c}.png) looking only near where the pose predicts the
fingertip, keep detections that agree, re-solve, repeat. The lens is fixed at --f (a C270: ~1420 px) unless --free-f.

    python calib_tools/solve_iterative.py cam0 sweep_zone13,sweep_zone14 [--f 1420] [--free-f] [--write]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

Y = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Y))
from calib3d import wedge_tip  # noqa: E402

key, dirs = sys.argv[1], sys.argv[2].split(",")
f0 = float(sys.argv[sys.argv.index("--f") + 1]) if "--f" in sys.argv else 1420.0
free_f, write = "--free-f" in sys.argv, "--write" in sys.argv
W, H = 1280, 720
recs = [(d, r) for d in dirs for r in json.loads((Y / "captures" / d / "sweep.json").read_text())]
P = np.array([r["fk"] for _, r in recs], float)
frames = []
for d, r in recs:
    fr = [cv2.imread(str(Y / "captures" / d / f"{r['i']:02d}_{key}_{t}.png")) for t in ("a", "a2", "b", "c")]
    frames.append(fr)
K = np.array([[f0, 0, W / 2], [0, f0, H / 2], [0, 0, 1]])
dist = np.zeros(5)
# the rough start: RANSAC over the original detections
I0 = np.array([r["px"].get(key) or [np.nan, np.nan] for _, r in recs], float)
ok = ~np.isnan(I0[:, 0])
_, rv, tv, inl = cv2.solvePnPRansac(P[ok], I0[ok], K, dist, reprojectionError=10.0, iterationsCount=6000, confidence=0.999)
for it in range(5):
    pred = cv2.projectPoints(P, rv, tv, K, dist)[0].reshape(-1, 2)
    I = np.full((len(P), 2), np.nan)
    for j, (fa, fa2, fb, fc) in enumerate(frames):
        if fa is None or not (0 <= pred[j, 0] < W and 0 <= pred[j, 1] < H):
            continue
        uv, _ = wedge_tip(fa, fb, fc, fa2, near=(float(pred[j, 0]), float(pred[j, 1])))
        if uv is not None and np.hypot(uv[0] - pred[j, 0], uv[1] - pred[j, 1]) < (40 if it < 2 else 20):
            I[j] = uv
    use = ~np.isnan(I[:, 0])
    if use.sum() < 6:
        sys.exit(f"{key}: only {use.sum()} detections agree with the pose")
    if free_f and use.sum() >= 12:
        fl = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
              | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
        _, K, dist, rvs, tvs = cv2.calibrateCamera([P[use].astype(np.float32)], [I[use].astype(np.float32)], (W, H), K.copy(), None, flags=fl)
        rv, tv, dist = rvs[0], tvs[0], dist.ravel()
    else:
        rv, tv = cv2.solvePnPRefineLM(P[use], I[use], K, dist, rv, tv)
    e = np.linalg.norm(cv2.projectPoints(P[use], rv, tv, K, dist)[0].reshape(-1, 2) - I[use], axis=1)
    C = (-cv2.Rodrigues(rv)[0].T @ np.ravel(tv))
    print(f"{key} pass {it + 1}: {use.sum()}/{len(P)} fingertips, rms {np.sqrt(np.mean(e ** 2)):.1f} px, "
          f"f={K[0, 0]:.0f}, k1={dist[0]:+.3f}, camera at {np.round(C, 3)}")
if write:
    cams = json.loads((Y / "cameras.json").read_text())
    R = cv2.Rodrigues(rv)[0]
    cams[key] = {"K": K.tolist(), "dist": list(np.ravel(dist)), "R": R.tolist(), "t": np.ravel(tv).tolist(), "size": [W, H],
                 "rms_px": float(np.sqrt(np.mean(e ** 2))), "center": C.tolist(), "n_points": int(use.sum()),
                 "fov_deg": float(2 * np.degrees(np.arctan(W / 2 / K[0, 0]))),
                 "method": f"iterative re-detection near the predicted tip ({', '.join(dirs)}), f {'free' if free_f else 'fixed'}"}
    (Y / "cameras.json").write_text(json.dumps(cams, indent=1))
    print(f"wrote {key} into cameras.json")
