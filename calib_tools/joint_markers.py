"""All cameras at once: poses (lens kept) from a flat ArUco sheet seen by every camera + the fingertip sweep.

    python captures/calib_debug/joint_markers.py [--write]

Unknowns: each camera's pose; each marker's (x, y, angle) on the table (z = 0) and ONE marker size shared by
all. Residuals: every detected marker corner in every camera (1 px), and every fingertip detection of
sweep_zone8 whose 3D position the arm's FK gives (robust, 5 px - the detector is biased from some angles).
The markers tie the cameras to each other; the fingertips tie them all to the robot frame. Markers seen by
one camera only carry no information and are left out.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

Y = Path("/Users/solotech/yam")
sys.path.insert(0, str(Y))
from calib3d import CameraModel  # noqa: E402

write = "--write" in sys.argv
cams = json.loads((Y / "cameras.json").read_text())
keys = sorted(cams)
p = cv2.aruco.DetectorParameters()
p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax, p.adaptiveThreshWinSizeStep = 3, 53, 4
p.minMarkerPerimeterRate, p.polygonalApproxAccuracyRate = 0.01, 0.05
p.cornerRefinementMethod, p.minCornerDistanceRate, p.errorCorrectionRate = cv2.aruco.CORNER_REFINE_SUBPIX, 0.02, 0.8
det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000), p)

obs = {}  # (key, id) -> 4x2 corners
for k in keys:
    g = cv2.cvtColor(cv2.imread(str(Y / f"captures/calib_debug/mk_{k}.png")), cv2.COLOR_BGR2GRAY)
    for sc in (1, 2, 3):
        gg = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC) if sc > 1 else g
        c, ids, _ = det.detectMarkers(gg)
        for cc, i in zip(c or [], [] if ids is None else ids.ravel()):
            obs.setdefault((k, int(i)), cc.reshape(4, 2) / sc)  # the first (native) scale that found it wins
ids = sorted({i for (_, i) in obs if sum((k, i) in obs for k in keys) >= 2})
print("markers seen by 2+ cameras:", len(ids), "|", {k: sum((k, i) in obs for i in ids) for k in keys})

sweep = json.loads((Y / "captures/sweep_zone8/sweep.json").read_text())
tips = [(k, np.array(r["fk"]), np.array(r["px"][k])) for r in sweep for k in keys if r["px"].get(k)]

# initial state: poses from cameras.json; markers from the camera that sees each best
def cam_of(x, k):
    rv, tv = x[:3], x[3:6]
    return rv, tv
models = {k: CameraModel(cams[k]) for k in keys}
K = {k: np.array(cams[k]["K"]) for k in keys}
D = {k: np.array(cams[k]["dist"]) for k in keys}
ref = "cam1"  # the reference for the markers' starting positions: calibrated on the sheet corners at z = 0
# markers seen by the reference start from it; the rest from whichever camera sees them largest
init = {}
for i in ids:
    k = ref if (ref, i) in obs else max((kk for kk in keys if (kk, i) in obs),
                                        key=lambda kk: cv2.contourArea(obs[(kk, i)].astype(np.float32)))
    init[i] = np.array([models[k].hit_plane(u, v, 0.0)[:2] for u, v in obs[(k, i)]])
# the corner order on the table (the printed sheet faces up, so one handedness for all)
hand = np.sign(np.median([np.cross(P[1] - P[0], P[2] - P[1]) for P in init.values()]))
unit = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) * [1, hand]
print("corner handedness on the table:", int(hand))
# each camera's starting pose: PnP on the markers' starting corners (not the old, inconsistent poses)
x0 = []
for k in keys:
    O = np.vstack([np.c_[init[i], np.zeros(4)] for i in ids if (k, i) in obs])
    I = np.vstack([obs[(k, i)] for i in ids if (k, i) in obs])
    R0 = np.array(cams[k]["R"]); t0 = np.array(cams[k]["t"]).reshape(3, 1)
    # start from the fingertip-based pose already in cameras.json: a planar target alone has a mirror-image
    # pose that fits the markers just as well (cam2 flipped to it), the arm's 3D fingertips do not
    x0 += list(cv2.Rodrigues(R0)[0].ravel()) + list(t0.ravel())
sizes = []
for i in ids:
    P = init[i]
    x0 += list(P.mean(axis=0)) + [float(np.arctan2(*(P[1] - P[0])[::-1]))]
    sizes.append(np.mean(np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)))
x0.append(float(np.log(np.median(sizes))))
x0 = np.array(x0)
print(f"starting marker size {np.median(sizes) * 100:.2f} cm")
nc = len(keys)


def corners3d(x, j):
    cx, cy, a = x[6 * nc + 3 * j: 6 * nc + 3 * j + 3]
    s = np.exp(x[-1])  # log-size: always positive
    Rz = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return np.c_[(unit * s) @ Rz.T + [cx, cy], np.zeros(4)]


def resid(x):
    out = []
    for ci, k in enumerate(keys):
        rv, tv = x[6 * ci:6 * ci + 3], x[6 * ci + 3:6 * ci + 6]
        for j, i in enumerate(ids):
            if (k, i) in obs:
                pr = cv2.projectPoints(corners3d(x, j), rv, tv, K[k], D[k])[0].reshape(-1, 2)
                out.append(((pr - obs[(k, i)]) / 1.0).ravel())
        T = [(P, uv) for kk, P, uv in tips if kk == k] if TIPS_ON else []
        if T:
            pr = cv2.projectPoints(np.array([P for P, _ in T]), rv, tv, K[k], D[k])[0].reshape(-1, 2)
            out.append(((pr - np.array([uv for _, uv in T])) / 3.0).ravel())
    return np.concatenate(out)


# stage 1: markers only, cam1 held fixed (it defines the frame on the table): cams 0/2 + markers + size
TIPS_ON = False
fix = keys.index(ref)
free = np.ones(len(x0), bool); free[6 * fix:6 * fix + 6] = False
def r_free(xf):
    x = x0.copy(); x[free] = xf
    return resid(x)
for rnd in range(4):  # solve, then drop markers that fit badly in any camera (misread IDs), and repeat
    sol = least_squares(r_free, x0[free], loss="soft_l1", f_scale=2.0, max_nfev=600)
    x1 = x0.copy(); x1[free] = sol.x
    bad = set()
    for ci, k in enumerate(keys):
        rv, tv = x1[6 * ci:6 * ci + 3], x1[6 * ci + 3:6 * ci + 6]
        for j, i in enumerate(ids):
            if (k, i) in obs and np.linalg.norm(cv2.projectPoints(corners3d(x1, j), rv, tv, K[k], D[k])[0].reshape(-1, 2)
                                                - obs[(k, i)], axis=1).max() > 10:
                bad.add(i)
    if not bad:
        break
    keepj = [j for j, i in enumerate(ids) if i not in bad]
    print(f"   round {rnd}: dropping {len(bad)} badly fitting marker(s) {sorted(bad)}")
    x0 = np.r_[x1[:6 * nc], np.concatenate([x1[6 * nc + 3 * j:6 * nc + 3 * j + 3] for j in keepj]), x1[-1]]
    ids = [i for i in ids if i not in bad]
    free = np.ones(len(x0), bool); free[6 * fix:6 * fix + 6] = False
print(f"stage 1 (markers only, {ref} fixed): marker size {np.exp(x1[-1]) * 100:.2f} cm")
for ci, k in enumerate(keys):
    rv, tv = x1[6 * ci:6 * ci + 3], x1[6 * ci + 3:6 * ci + 6]
    parts = [np.linalg.norm(cv2.projectPoints(corners3d(x1, j), rv, tv, K[k], D[k])[0].reshape(-1, 2) - obs[(k, i)], axis=1)
             for j, i in enumerate(ids) if (k, i) in obs]
    if not parts:
        print(f"   {k}: no markers left - its pose stays the fingertip one"); continue
    em = np.concatenate(parts)
    print(f"   {k}: marker corners rms {np.sqrt(np.mean(em ** 2)):.2f} px, median {np.median(em):.2f} px ({len(parts)} markers)")
# stage 2: all cameras free, with the fingertips that agree with stage 1 (within 12 px) as the robot-frame anchor
keep = []
for kk, P, uv in tips:
    ci = keys.index(kk)
    pr = cv2.projectPoints(P[None], x1[6 * ci:6 * ci + 3], x1[6 * ci + 3:6 * ci + 6], K[kk], D[kk])[0].ravel()
    if np.linalg.norm(pr - uv) < 12:
        keep.append((kk, P, uv))
print(f"stage 2 anchors: {len(keep)} fingertip detections agree with stage 1 "
      f"({ {k: sum(1 for kk, _, _ in keep if kk == k) for k in keys} })")
tips[:] = keep
TIPS_ON = True
sol = least_squares(resid, x1, loss="soft_l1", f_scale=2.0, max_nfev=600) if len(keep) >= 6 else None
x = sol.x if sol is not None else x1
print(f"final marker size {np.exp(x[-1]) * 100:.2f} cm")
report = {}
for ci, k in enumerate(keys):
    rv, tv = x[6 * ci:6 * ci + 3], x[6 * ci + 3:6 * ci + 6]
    em = [np.linalg.norm(cv2.projectPoints(corners3d(x, j), rv, tv, K[k], D[k])[0].reshape(-1, 2) - obs[(k, i)], axis=1)
          for j, i in enumerate(ids) if (k, i) in obs]
    em = np.concatenate(em) if em else np.array([])
    T = [(P, uv) for kk, P, uv in tips if kk == k]
    et = np.linalg.norm(cv2.projectPoints(np.array([P for P, _ in T]), rv, tv, K[k], D[k])[0].reshape(-1, 2)
                        - np.array([uv for _, uv in T]), axis=1) if T else np.array([])
    R = cv2.Rodrigues(rv)[0]
    C = (-R.T @ tv).ravel()
    moved = np.linalg.norm(C - np.array(cams[k]["center"])) * 100
    print(f"{k}: markers {len(em) // 4} rms {np.sqrt(np.mean(em ** 2)) if len(em) else float('nan'):.2f} px | fingertips "
          f"median {np.median(et) if len(et) else float('nan'):.1f} px ({(et < 12).sum()}/{len(et)} within 12 px) | camera at "
          f"{np.round(C, 3)} ({moved:.1f} cm from before)")
    report[k] = (R, tv, C, em, et)
if write:
    for k, (R, tv, C, em, et) in report.items():
        cams[k].update(R=R.tolist(), t=tv.ravel().tolist(), center=C.tolist(),
                       rms_px=float(np.sqrt(np.mean(em ** 2))) if len(em) else cams[k].get("rms_px"),
                       note="2026-09-23: joint solve - flat ArUco sheet in every camera + sweep_zone8 fingertips (lens kept)")
    (Y / "cameras.json").write_text(json.dumps(cams, indent=1))
    print("wrote cameras.json")


# ---- stage 3: the camera pair (consistent through the markers) -> the robot frame, 3D to 3D ----------------
from calib3d import triangulate
pair = [k for k in keys if k != "cam0"]
mod = {}
for ci, k in enumerate(keys):
    if k in pair:
        d = dict(cams[k]); d["R"] = cv2.Rodrigues(x1[6 * ci:6 * ci + 3])[0].tolist(); d["t"] = x1[6 * ci + 3:6 * ci + 6].tolist()
        mod[k] = CameraModel(d)
Pc, Pr = [], []
for r in json.loads((Y / "captures/sweep_zone8/sweep.json").read_text()):
    if all(r["px"].get(k) for k in pair):
        X, gap = triangulate(mod[pair[0]], r["px"][pair[0]], mod[pair[1]], r["px"][pair[1]])
        Pc.append(X); Pr.append(r["fk"]); print(f"   pt {r['i']:2d}: fk {np.round(r['fk'], 3)} cams {np.round(X, 3)} ray gap {gap * 100:.1f} cm")
Pc, Pr = np.array(Pc), np.array(Pr)
def rigid(A, B):  # B ~ R A + t
    ca, cb = A.mean(0), B.mean(0); U, S_, Vt = np.linalg.svd((A - ca).T @ (B - cb)); Dm = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ Dm @ U.T; return R, cb - R @ ca
if len(Pc) >= 4:
    use = np.ones(len(Pc), bool)
    for _ in range(3):
        R_, t_ = rigid(Pc[use], Pr[use]); e = np.linalg.norm((Pc @ R_.T + t_) - Pr, axis=1)
        use = e < max(0.01, 2.5 * np.median(e))
    ang = np.degrees(np.arccos(np.clip((np.trace(R_) - 1) / 2, -1, 1)))
    print(f"stage 3: {len(Pc)} points both cameras saw, {use.sum()} consistent; camera frame -> robot frame: "
          f"rotate {ang:.2f} deg, shift {np.round(t_ * 100, 2)} cm; residual {np.median(e[use]) * 1000:.1f} mm median")
    np.save(str(Y / "captures/calib_debug/frame_fix.npy"), np.c_[R_, t_])
