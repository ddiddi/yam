"""Re-aim a bumped camera: a knock on its mount mostly turns the camera, and a pure turn maps the old view onto the
new one whatever the depth. ORB matches between its empty-zone background (captures/bg_camN.png, taken when the
calibration held) and a live frame give that turn (RANSAC + Kabsch on the undistorted bearings); the lens and the
camera centre are kept, R becomes R_turn @ R_old. Nothing in the scene has to be cleared.

    YAM_CAM_MAP=1:2,2:1 python calib_tools/cam_rebump.py --cam cam1            # report only
    YAM_CAM_MAP=1:2,2:1 python calib_tools/cam_rebump.py --cam cam1 --write    # cameras.json (+ the background)
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tyro

Y = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Y))
from scene3d import BACKGROUND, grab  # noqa: E402


@dataclass
class Args:
    cam: str = "cam1"
    write: bool = False
    """Write the new R into cameras.json and re-register the background (the old one is kept as .bak)."""


def bearings(pts: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    n = cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), K, dist).reshape(-1, 2)
    b = np.c_[n, np.ones(len(n))]
    return b / np.linalg.norm(b, axis=1, keepdims=True)


def kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation R with R @ a_i ~ b_i."""
    U, _, Vt = np.linalg.svd(a.T @ b)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def main(a: Args) -> None:
    cams = json.loads((Y / "cameras.json").read_text())
    c = cams[a.cam]
    K, dist, R0 = (np.array(c[k], dtype=np.float64) for k in ("K", "dist", "R"))
    t0 = np.array(c["t"], dtype=np.float64)
    C = -R0.T @ t0
    bg = cv2.imread(str(BACKGROUND[a.cam]))
    fr = grab(int(a.cam[3:]))
    orb = cv2.ORB_create(5000)
    ka, da = orb.detectAndCompute(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), None)
    kb, db = orb.detectAndCompute(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), None)
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    m = sorted(m, key=lambda x: x.distance)[:1500]
    pa = np.array([ka[x.queryIdx].pt for x in m])
    pb = np.array([kb[x.trainIdx].pt for x in m])
    A, B = bearings(pa, K, dist), bearings(pb, K, dist)
    rng = np.random.default_rng(0)
    best, tol = None, np.radians(0.25)  # ~4 px at this focal length
    for _ in range(2000):
        i = rng.choice(len(A), 2, replace=False)
        Rt = kabsch(A[i], B[i])
        ang = np.arccos(np.clip(np.sum((A @ Rt.T) * B, axis=1), -1, 1))
        inl = ang < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    Rt = kabsch(A[best], B[best])
    ang = np.degrees(np.arccos(np.clip(np.sum((A[best] @ Rt.T) * B[best], axis=1), -1, 1)))
    turn = np.degrees(np.linalg.norm(cv2.Rodrigues(Rt)[0]))
    px = np.median(np.linalg.norm(pb[best] - pa[best], axis=1))
    print(f"{a.cam}: {best.sum()}/{len(A)} matches fit a pure turn of {turn:.2f} deg (median shift {px:.1f} px), "
          f"residual median {np.median(ang) * K[0, 0] * np.pi / 180:.2f} px, 90% {np.percentile(ang, 90) * K[0, 0] * np.pi / 180:.2f} px")
    R1 = Rt @ R0
    t1 = -R1 @ C
    if a.write:
        c["R"], c["t"] = R1.tolist(), t1.tolist()
        c["note"] = (c.get("note", "") + f" | {time.strftime('%Y-%m-%d')}: re-aimed after a bump "
                     f"({turn:.2f} deg turn from {best.sum()} background matches, calib_tools/cam_rebump.py)")
        (Y / "cameras.json").write_text(json.dumps(cams, indent=2))
        bp = Path(BACKGROUND[a.cam])
        bp.with_suffix(".png.bak").write_bytes(bp.read_bytes())
        # the background, turned with the camera, so the drift check measures from the new aim
        H = K @ Rt @ np.linalg.inv(K)
        cv2.imwrite(str(bp), cv2.warpPerspective(bg, H, (bg.shape[1], bg.shape[0]), borderMode=cv2.BORDER_REPLICATE))
        print(f"wrote cameras.json[{a.cam}] and {bp.name} (old one kept as {bp.name}.bak)")


if __name__ == "__main__":
    main(tyro.cli(Args))
