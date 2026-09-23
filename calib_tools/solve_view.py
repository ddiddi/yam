"""Pose of one camera (lens kept) from a fingertip sweep + the four taped-sheet corners (zone.json).

    python captures/calib_debug/solve_view.py cam0 captures/cameras_with_old_cam0.json sweep_zone8 [--write]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

Y = Path("/Users/solotech/yam")
sys.path.insert(0, str(Y))
import pick_place as pp  # noqa: E402

key, lens_file, sweep = sys.argv[1], sys.argv[2], sys.argv[3]
write = "--write" in sys.argv
lens = json.loads((Y / lens_file).read_text())[key]
K, dist = np.array(lens["K"]), np.array(lens["dist"])
recs = json.loads((Y / "captures" / sweep / "sweep.json").read_text())
tips = [(r["fk"], r["px"][key]) for r in recs if r["px"].get(key) is not None]
T_obj = np.array([p for p, _ in tips], np.float64)
T_img = np.array([u for _, u in tips], np.float64)
print(f"{key}: {len(tips)} fingertip detections, lens f={K[0, 0]:.0f}px dist {np.round(dist[:2], 3)}")
if len(tips) < 6:
    sys.exit(f"{key}: too few detections")
ok, rv, tv, inl = cv2.solvePnPRansac(T_obj, T_img, K, dist, reprojectionError=12.0, iterationsCount=5000)
if not ok or inl is None or len(inl) < 5:
    sys.exit(f"{key}: no consistent pose from the fingertips")
inl = inl.ravel()
rv, tv = cv2.solvePnPRefineLM(T_obj[inl], T_img[inl], K, dist, rv, tv)
print(f"  fingertip RANSAC: {len(inl)}/{len(tips)} inliers")

# the sheet corners in this view, matched to zone.json by the fingertip pose
from scene3d import grab
f = grab(int(key[3:]))  # a fresh frame: the arm is parked, the sheet is unobstructed (sweep frames have the arm)
Z = pp.load_zone()
Zo = np.c_[Z, np.zeros(len(Z))]
proj = cv2.projectPoints(Zo, rv, tv, K, dist)[0].reshape(-1, 2)
hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
roi = np.zeros(f.shape[:2], np.uint8)
cv2.fillPoly(roi, [proj.astype(np.int32)], 255)
roi = cv2.dilate(roi, np.ones((151, 151), np.uint8))
white = ((hsv[..., 1] < 60) & (hsv[..., 2] > 110) & (roi > 0)).astype(np.uint8) * 255  # greyer from some angles
white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
use_c = False
if cnts:
    c = max(cnts, key=cv2.contourArea)
    for eps in (0.01, 0.02, 0.03, 0.05):
        quad = cv2.approxPolyDP(c, eps * cv2.arcLength(c, True), True).reshape(-1, 2).astype(np.float32)
        if len(quad) == 4:
            break
    if len(quad) == 4:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        q = cv2.cornerSubPix(g, quad.reshape(-1, 1, 2), (7, 7), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)).reshape(-1, 2)
        order = [int(np.argmin(np.linalg.norm(q - p, axis=1))) for p in proj]
        if len(set(order)) == 4 and max(np.linalg.norm(q[order] - proj, axis=1)) < 120:
            C_img = q[order].astype(np.float64)
            use_c = True
            print(f"  sheet corners found, {np.round(np.linalg.norm(C_img - proj, axis=1), 1)} px from the fingertip pose")
if not use_c:
    print("  sheet corners NOT found cleanly - fingertips only")
for it in range(4):
    e_t = np.linalg.norm(cv2.projectPoints(T_obj, rv, tv, K, dist)[0].reshape(-1, 2) - T_img, axis=1)
    use = e_t <= (20 if it == 0 else 10)
    O = np.vstack([Zo, T_obj[use]]) if use_c else T_obj[use]
    I = np.vstack([C_img, T_img[use]]) if use_c else T_img[use]
    ok, rv, tv = cv2.solvePnP(O, I, K, dist, rv, tv, useExtrinsicGuess=True)
e_t = np.linalg.norm(cv2.projectPoints(T_obj, rv, tv, K, dist)[0].reshape(-1, 2) - T_img, axis=1)
e_c = np.linalg.norm(cv2.projectPoints(Zo, rv, tv, K, dist)[0].reshape(-1, 2) - C_img, axis=1) if use_c else np.array([])
R = cv2.Rodrigues(rv)[0]
C = (-R.T @ tv).ravel()
print(f"  final: corners {np.round(e_c, 1)} px, fingertips {int(use.sum())} used rms "
      f"{np.sqrt(np.mean(e_t[use] ** 2)):.1f} px, camera at {np.round(C, 3)}")
v = f.copy()
cv2.polylines(v, [cv2.projectPoints(Zo, rv, tv, K, dist)[0].reshape(-1, 2).astype(np.int32)], True, (0, 0, 255), 2)
for r in recs:
    p = cv2.projectPoints(np.array([r["fk"]], np.float64), rv, tv, K, dist)[0].ravel()
    cv2.circle(v, (int(p[0]), int(p[1])), 4, (255, 0, 255), -1)
cv2.imwrite(str(Y / "captures/calib_debug" / f"check_{key}.jpg"), cv2.resize(v, (960, 540)))
if write:
    cams = json.loads((Y / "cameras.json").read_text())
    cam = dict(lens)
    cam.update(R=R.tolist(), t=tv.ravel().tolist(), center=C.tolist(),
               rms_px=float(np.sqrt(np.mean(np.r_[e_c, e_t[use]] ** 2))), n_points=int(use.sum() + (4 if use_c else 0)),
               note=f"2026-09-23: pose from {'sheet corners + ' if use_c else ''}{sweep} fingertips (PnP, lens kept)")
    cams[key] = cam
    (Y / "cameras.json").write_text(json.dumps(cams, indent=1))
    print(f"  wrote {key} into cameras.json")
