"""Weigh-station calibration: the 11x8 ChArUco sheet (legacy layout, 15 mm squares, DICT_4X4_50) lying on
the scale top at a hand-measured height puts cam0 / cam2 in the robot frame.

    YAM_CAM_MAP=1:2,2:1 python calib_tools/station_calib.py --top-z 0.052            # report only
    YAM_CAM_MAP=1:2,2:1 python calib_tools/station_calib.py --top-z 0.052 --write    # cameras.json + station.json

1. cam1 (already calibrated) back-projects the markers it reads onto z = top_z; a rigid 2D fit (the square
   size is known, so no scale) gives the board's pose on the scale. Misread markers are RANSAC outliers.
2. Every other camera keeps its lens and gets its pose by PnP on the ChArUco corners it sees (plus its marker
   corners), with a third held out to check the fit.
3. station.json: the board outline and the scale platform (the board is centred on it by hand, so the
   platform outline is measured separately - see --platform) at top_z.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tyro

Y = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Y))
from calib3d import CameraModel  # noqa: E402
from scene3d import grab  # noqa: E402

SQUARES = (11, 8)
SQUARE = 0.015
MARKER = 0.75 * SQUARE  # only the layout matters for the ids; corners use the true marker size below
DICT = cv2.aruco.DICT_4X4_50


@dataclass
class Args:
    top_z: float = 0.052
    """Height of the board surface (the scale top) above the table, m (measured by hand)."""
    ref: str = "cam1"
    cams: str = "cam0,cam2"
    write: bool = False
    marker_ratio: float = 0.75
    """Printed marker side / square side (the sheet's generator setting)."""


def board(ratio: float):
    d = cv2.aruco.getPredefinedDictionary(DICT)
    b = cv2.aruco.CharucoBoard(SQUARES, SQUARE, ratio * SQUARE, d)
    b.setLegacyPattern(True)
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return b, cv2.aruco.ArucoDetector(d, p), cv2.aruco.CharucoDetector(b, cv2.aruco.CharucoParameters(), p)


def mirror(pts: np.ndarray) -> np.ndarray:
    """Board frame -> a right-handed frame seen from above: the board's own z points into the sheet."""
    out = np.array(pts, dtype=np.float64)[..., :2].copy()
    out[..., 1] *= -1
    return out


def rigid2d(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rotation + translation (no scale) src -> dst."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - ms).T @ (dst - md))
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    return R, md - R @ ms


def main(a: Args) -> None:
    raw = json.loads((Y / "cameras.json").read_text())
    b, mdet, cdet = board(a.marker_ratio)
    ids_all = list(b.getIds().ravel())
    objp = b.getObjPoints()
    frames = {k: grab(int(k[3:])) for k in [a.ref] + a.cams.split(",")}
    for k, f in frames.items():
        cv2.imwrite(str(Y / "captures" / f"station_{k}.png"), f)

    # 1. board pose from the reference camera
    ref = CameraModel(raw[a.ref])
    c, ids, _ = mdet.detectMarkers(cv2.cvtColor(frames[a.ref], cv2.COLOR_BGR2GRAY))
    if ids is None:
        raise SystemExit(f"{a.ref} reads no markers")
    per = []  # (marker id, board xy (4,2), world xy (4,2))
    for i, cc in zip(ids.ravel(), c):
        if int(i) not in ids_all:
            continue
        w = np.array([ref.hit_plane(float(u), float(v), a.top_z)[:2] for u, v in cc.reshape(4, 2)])
        per.append((int(i), mirror(objp[ids_all.index(int(i))]), w))
    best = None
    for i0, _, _ in per:  # RANSAC over markers: a misread id moves all four of its corners together
        for i1, _, _ in per:
            if i1 <= i0:
                continue
            sub = [q for q in per if q[0] in (i0, i1)]
            R, t = rigid2d(np.vstack([q[1] for q in sub]), np.vstack([q[2] for q in sub]))
            good = [q for q in per if np.linalg.norm(q[1] @ R.T + t - q[2], axis=1).max() < 0.005]
            if best is None or len(good) > len(best):
                best = good
    if best is None or len(best) < 3:
        raise SystemExit(f"{a.ref}: fewer than 3 consistent markers - the board cannot be placed")
    R, t = rigid2d(np.vstack([q[1] for q in best]), np.vstack([q[2] for q in best]))
    res = np.hstack([np.linalg.norm(q[1] @ R.T + t - q[2], axis=1) for q in best])
    print(f"{a.ref}: board placed from markers {[q[0] for q in best]} (dropped {sorted({q[0] for q in per} - {q[0] for q in best})}), "
          f"rms {np.sqrt(np.mean(res ** 2)) * 1000:.1f} mm, yaw {np.degrees(np.arctan2(R[1, 0], R[0, 0])):.1f} deg")

    def to_world(bp: np.ndarray) -> np.ndarray:
        xy = mirror(bp) @ R.T + t
        return np.c_[xy, np.full(len(xy), a.top_z)]

    # 2. each camera from the board
    for k in a.cams.split(","):
        g = cv2.cvtColor(frames[k], cv2.COLOR_BGR2GRAY)
        cc, ci, mc, mi = cdet.detectBoard(g)
        O, I = [], []
        if ci is not None:
            O.append(to_world(np.asarray(b.getChessboardCorners())[ci.ravel()]))
            I.append(cc.reshape(-1, 2))
        if mi is not None:
            for i, m in zip(mi.ravel(), mc):
                if int(i) in ids_all:
                    O.append(to_world(objp[ids_all.index(int(i))]))
                    I.append(m.reshape(4, 2))
        if not O or sum(len(o) for o in O) < 12:
            print(f"{k}: {0 if not O else sum(len(o) for o in O)} board points - not calibrated")
            continue
        O, I = np.vstack(O), np.vstack(I).astype(np.float64)
        K, dist = np.array(raw[k]["K"]), np.array(raw[k]["dist"])
        hold = np.zeros(len(O), bool)
        hold[::3] = True
        ok, rv, tv, inl = cv2.solvePnPRansac(O[~hold], I[~hold], K, dist, reprojectionError=4.0, iterationsCount=5000)
        if not ok:
            print(f"{k}: no pose")
            continue
        fit = np.where(~hold)[0][inl.ravel()]
        rv, tv = cv2.solvePnPRefineLM(O[fit], I[fit], K, dist, rv, tv)
        e = np.linalg.norm(cv2.projectPoints(O, rv, tv, K, dist)[0].reshape(-1, 2) - I, axis=1)
        Rc = cv2.Rodrigues(rv)[0]
        C = (-Rc.T @ tv).ravel()
        e_fit, e_hold = e[fit], e[hold]
        print(f"{k}: {len(O)} board points ({len(fit)} fitted, {hold.sum()} held out): fit {np.sqrt(np.mean(e_fit ** 2)):.2f} px, "
              f"held-out median {np.median(e_hold):.2f} px, camera at {np.round(C, 3).tolist()} m")
        # the plane alone leaves the camera's distance poorly pinned for a steep view: say so, do not hide it
        if np.median(e_hold) > 4.0:
            print(f"   {k}: held-out error above 4 px - NOT saved")
            continue
        if a.write:
            cam = dict(raw[k])
            cam.update(R=Rc.tolist(), t=tv.ravel().tolist(), center=C.tolist(), rms_px=float(np.sqrt(np.mean(e_fit ** 2))),
                       n_points=int(len(fit)), note=f"pose from the ChArUco sheet on the scale (top {a.top_z * 100:.1f} cm), "
                       f"placed by {a.ref}; lens kept")
            raw[k] = cam

    # 3. station.json
    outline = to_world(np.array([[0, 0, 0], [SQUARES[0] * SQUARE, 0, 0], [SQUARES[0] * SQUARE, SQUARES[1] * SQUARE, 0],
                                 [0, SQUARES[1] * SQUARE, 0]]))
    station = {"top_z": a.top_z, "board_outline": outline[:, :2].round(4).tolist(),
               "board_centre": outline[:, :2].mean(0).round(4).tolist(),
               "board_yaw_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
               "placed_by": a.ref, "board": {"squares": SQUARES, "square_m": SQUARE, "legacy": True, "dict": "DICT_4X4_50"}}
    print(f"board centre {station['board_centre']} m, outline {station['board_outline']}")
    if a.write:
        (Y / "captures" / "cameras_before_station.json").write_text((Y / "cameras.json").read_text())
        (Y / "cameras.json").write_text(json.dumps(raw, indent=1))
        old = json.loads((Y / "station.json").read_text()) if (Y / "station.json").exists() else {}
        old.update(station)
        (Y / "station.json").write_text(json.dumps(old, indent=1))
        print("wrote cameras.json (previous in captures/cameras_before_station.json) and station.json")


if __name__ == "__main__":
    main(tyro.cli(Args))
