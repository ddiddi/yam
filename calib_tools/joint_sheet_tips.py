"""One camera + the ChArUco sheet's pose on the scale, solved together: the sheet's corners (many, exact) fix the lens
and the camera relative to the sheet; the fingertip sweep (FK in the robot frame) fixes where both are in the robot
frame. Unknowns: f, k1, camera pose (6), sheet x, y, yaw (it lies flat at --top-z). Robust (soft-L1) on the fingertips.

    python calib_tools/joint_sheet_tips.py SWEEP_KEY LIVE_INDEX sweep_zone13,sweep_zone14 [--top-z 0.052] [--write]
      SWEEP_KEY: the camera's name in the sweeps (e.g. cam0); LIVE_INDEX: its camN index now (YAM_CAM_MAP applies)
    --write: cameras.json[cam<LIVE_INDEX>] and the sheet/footprint/place in station.json
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
cv2.imwrite(str(Y / f"captures/sheet_live_cam{live}.png"), f)
g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
cc, ci, mc, mi = cdet.detectBoard(g)
Bp = [np.asarray(b.getChessboardCorners())[ci.ravel()]]; Bi = [cc.reshape(-1, 2)]
for i, m in zip(mi.ravel(), mc):
    if int(i) in ids_all:
        Bp.append(objp[ids_all.index(int(i))]); Bi.append(m.reshape(4, 2))
Bp = sc.mirror(np.vstack(Bp)); Bi = np.vstack(Bi).astype(float)
recs = [r for d in dirs for r in json.loads((Y / "captures" / d / "sweep.json").read_text()) if r["px"].get(key)]
Tp = np.array([r["fk"] for r in recs]); Ti = np.array([r["px"][key] for r in recs], float)
print(f"{len(Bi)} sheet points, {len(Tp)} fingertip detections")
# start: the sheet where station.json has it; the camera from PnP on it
A = sc.mirror(np.array([[0, 0, 0], [0.165, 0, 0], [0.165, 0.12, 0], [0, 0.12, 0]], float))
R2, t2 = sc.rigid2d(A, np.array(st["board_outline"]))
yaw0 = np.arctan2(R2[1, 0], R2[0, 0])


def sheet_world(p):
    c, s = np.cos(p[2]), np.sin(p[2])
    xy = Bp @ np.array([[c, -s], [s, c]]).T + p[:2]
    return np.c_[xy, np.full(len(xy), top)]


def cam_of(p):
    fx, k1 = p[3], p[4]
    K = np.array([[fx, 0, 640], [0, fx, 360], [0, 0, 1]]); d = np.array([k1, 0, 0, 0, 0.0])
    return K, d, p[5:8], p[8:11]


K0 = np.array([[1500.0, 0, 640], [0, 1500, 360], [0, 0, 1]])
_, rv0, tv0 = cv2.solvePnP(sheet_world([t2[0], t2[1], yaw0]), Bi, K0, np.zeros(5))
p0 = np.r_[t2, yaw0, 1500.0, 0.0, rv0.ravel(), tv0.ravel()]


def resid(p, w_tip=1.0):
    K, d, rv, tv = cam_of(p)
    e1 = (cv2.projectPoints(sheet_world(p), rv, tv, K, d)[0].reshape(-1, 2) - Bi).ravel()
    e2 = (cv2.projectPoints(Tp, rv, tv, K, d)[0].reshape(-1, 2) - Ti).ravel()
    return np.r_[e1, w_tip * e2 * np.sqrt(len(Bi) / max(len(Tp), 1)) * 0.5]


s = least_squares(resid, p0, loss="soft_l1", f_scale=5.0, x_scale="jac")
p = s.x
K, d, rv, tv = cam_of(p)
eb = np.linalg.norm(cv2.projectPoints(sheet_world(p), rv, tv, K, d)[0].reshape(-1, 2) - Bi, axis=1)
et = np.linalg.norm(cv2.projectPoints(Tp, rv, tv, K, d)[0].reshape(-1, 2) - Ti, axis=1)
R = cv2.Rodrigues(rv)[0]; C = (-R.T @ tv).ravel()
move = p[:2] - t2
print(f"f={p[3]:.0f}px k1={p[4]:+.3f}; sheet rms {np.sqrt(np.mean(eb ** 2)):.2f} px; fingertips median {np.median(et):.1f} px, "
      f"{(et < 10).sum()}/{len(et)} within 10 px; camera at {np.round(C, 3)}")
print(f"the sheet moved by {np.round(move, 4)} m ({np.linalg.norm(move) * 100:.1f} cm) and turned {np.degrees(p[2] - yaw0):+.2f} deg "
      "since station.json")
if write:
    cams = json.loads((Y / "cameras.json").read_text())
    cams[f"cam{live}"] = {"K": K.tolist(), "dist": d.tolist(), "R": R.tolist(), "t": np.ravel(tv).tolist(), "size": [1280, 720],
                          "rms_px": float(np.sqrt(np.mean(eb ** 2))), "center": C.tolist(), "n_points": int(len(Bi) + (et < 10).sum()),
                          "fov_deg": float(2 * np.degrees(np.arctan(640 / p[3]))),
                          "method": f"joint: ChArUco sheet corners + fingertip sweep ({', '.join(dirs)}, detections of {key}); sheet pose solved"}
    (Y / "cameras.json").write_text(json.dumps(cams, indent=1))
    # move the station with the sheet
    Rm = np.array([[np.cos(p[2] - yaw0), -np.sin(p[2] - yaw0)], [np.sin(p[2] - yaw0), np.cos(p[2] - yaw0)]])
    c_old = np.array(st["board_outline"]).mean(0)
    mv = lambda q: ((np.asarray(q) - c_old) @ Rm.T + c_old + move).round(4).tolist()
    (Y / "captures/station_before_4cam.json").write_text(json.dumps(st, indent=1))
    st.update(board_outline=mv(st["board_outline"]), board_centre=mv(st["board_centre"]), footprint=mv(st["footprint"]),
              place=mv(st["place"]), board_yaw_deg=float(np.degrees(p[2])), placed_by=f"cam{live} joint sheet+fingertips")
    (Y / "station.json").write_text(json.dumps(st, indent=1))
    print(f"wrote cameras.json[cam{live}] and station.json (old one in captures/station_before_4cam.json)")
