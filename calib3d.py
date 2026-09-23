"""Full camera calibration of both cameras against the YAM arm, and 3D localisation from them.

    python calib3d.py --sweep          # ARM MOVES, table must be clear: fingertips visit a 3x3x3 grid,
                                       #   both cameras record the gripper blink -> captures/sweep/*.png + sweep.json
    python calib3d.py --solve          # offline: intrinsics + distortion + pose of each camera in the robot frame
                                       #   -> cameras.json  (re-run any time, e.g. after improving the detector)
    python calib3d.py --check          # offline: reprojection report per point

Camera model per camera (cameras.json): K (3x3), dist (k1 k2 p1 p2 k3), R (3x3), t (3): a robot-frame point
X projects as  x = K * distort(R X + t).  The robot frame is the arm base: x forward, z up, table at z ~ 0.

The calibration target is the closed fingertip (grasp_site) whose 3D position comes from the arm's FK.
Detection: frames a (open) -> b (closed) -> c (open); the finger wedge is the blob that is dark in b only,
which rejects people moving in the background and the fingers' own shadow.
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

HERE = Path(__file__).resolve().parent
SWEEPS = [HERE / "captures" / "sweep", HERE / "captures" / "sweep_low"]  # all are used by --solve
CAMERAS = HERE / "cameras.json"

GRID_X = [0.22, 0.32, 0.42]
GRID_Y = [-0.28, -0.16, -0.04]
GRIDS_Z = {"full": [0.03, 0.10, 0.18], "low": [0.025, 0.06, 0.10]}  # cam 1 sits at table level: sees z <~ 0.10


@dataclass
class Args:
    sweep: bool = False
    grid: str = "full"
    """'full' (3..18 cm, -> captures/sweep) or 'low' (2.5..10 cm, -> captures/sweep_low)."""
    xy: str | None = None
    """Restrict the sweep to these xy columns, e.g. "0.22,-0.04;0.42,-0.28" (an object is on the table);
    each column is visited top-down and the arm travels between columns at Z_TRAVEL."""
    gx: str | None = None
    """Override the grid's x values (m), e.g. "0.18,0.28,0.38" - aim the sweep at the workspace zone."""
    gy: str | None = None
    """Override the grid's y values (m), e.g. "-0.16,-0.05,0.06"."""
    dirs: str = "sweep,sweep_low"
    """Sweep folders under captures/: --sweep writes the first, --solve/--check read all of them. After a
    camera moves, sweep into a new folder and solve from it alone (old sweeps saw the old camera pose)."""
    board: bool = False
    """Camera only: calibrate --board-cams against the printed ChArUco board (charuco_board.json) lying on the
    table, located in the robot frame by the cameras already in cameras.json."""
    markers: bool = False
    """Camera only: calibrate --board-cams from ANY flat sheet of ArUco markers (DICT_4X4_1000) lying on the
    table - their corners are put in the robot frame by --ref, no board layout or print size needed."""
    ref: str = "cam1"
    """The calibrated camera that places the marker corners on the table plane (z = 0)."""
    board_cams: str = "cam2"
    """Which camera(s) to (re)calibrate from the board, e.g. "cam2" or "cam0,cam2"."""
    solve: bool = False
    check: bool = False
    cam: int = 0
    cam2: int = 1
    cams: str | None = None
    """All cameras to sweep, as OpenCV indices; camera N is saved as "camN", e.g. "0,1,2". Default: --cam, --cam2."""
    channel: str = "can0"
    hz: float = 100.0


# ------------------------------------------------------------------------------------------- detection
def wedge_tip(a: np.ndarray, b: np.ndarray, c: np.ndarray, a2: np.ndarray | None = None, dark: int = 70, min_area: float = 300.0):
    """Pixel of the closed fingertip: lowest rows of the biggest blob that is dark in b but not in a or c.
    With a second open frame a2 (taken shortly after a), anything that already changed between a and a2
    (people moving in the background) is rejected too."""
    ga, gb, gc = (cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in (a, b, c))
    m = ((gb < dark) & (ga >= dark) & (gc >= dark)).astype(np.uint8) * 255
    if a2 is not None:
        ga2 = cv2.cvtColor(a2, cv2.COLOR_BGR2GRAY)
        moving = cv2.dilate((cv2.absdiff(ga, ga2) > 20).astype(np.uint8) * 255, np.ones((15, 15), np.uint8))
        m = cv2.bitwise_and(m, cv2.bitwise_not(moving))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [k for k in cnts if cv2.contourArea(k) >= min_area]
    if not cnts:
        return None, m
    big = max(cnts, key=cv2.contourArea)
    keep = np.zeros_like(m)
    cv2.drawContours(keep, [big], -1, 255, -1)
    vs, us = np.where(keep > 0)
    h, w = m.shape
    if vs.max() >= h - 3 or us.min() <= 2 or us.max() >= w - 3:
        return None, keep  # wedge cut off by the frame edge: its tip is not visible
    sel = vs >= vs.max() - 4
    return (float(us[sel].mean()), float(vs[sel].mean())), keep


# ----------------------------------------------------------------------------------------------- sweep
def stage_sweep(args: Args) -> None:
    from lerobot_record import LiveCamera
    from pick_bottle import GRIP_OPEN, Z_TRAVEL, Arm, Planner, goto_joint, move_cartesian, settle

    dirs = sweep_dirs(args)
    SWEEP = dirs[0] if args.dirs != Args.dirs else (SWEEPS[0] if args.grid == "full" else SWEEPS[1])
    GRID_Z = GRIDS_Z[args.grid]
    GX = [float(v) for v in args.gx.split(",")] if args.gx else GRID_X
    GY = [float(v) for v in args.gy.split(",")] if args.gy else GRID_Y
    SWEEP.mkdir(parents=True, exist_ok=True)
    idx = [int(v) for v in args.cams.split(",")] if args.cams else [args.cam, args.cam2]
    cams = {f"cam{i}": LiveCamera(i) for i in idx}
    planner = Planner()
    arm = Arm(False, args.channel, args.hz)
    records = []
    pts = []
    yaw = tilt = None  # set by the first move; park() only lifts once the arm is out
    if args.xy:  # column mode: (x, y, Z_TRAVEL) marks "travel here first", then the z levels top-down
        for col in args.xy.split(";"):
            x, y = (float(v) for v in col.split(","))
            pts.append((x, y, Z_TRAVEL))
            pts += [(x, y, z) for z in sorted(GRID_Z, reverse=True)]
    else:  # snake through the grid so consecutive points are neighbours
        for iz, z in enumerate(GRID_Z):
            ys = GY if iz % 2 == 0 else GY[::-1]
            for iy, y in enumerate(ys):
                xs = GX if (iz + iy) % 2 == 0 else GX[::-1]
                for x in xs:
                    pts.append((x, y, z))
    try:
        arm.set_grip(GRIP_OPEN, 1.0)
        q0, yaw, tilt = planner.ik(pts[0][0], pts[0][1], Z_TRAVEL, arm.q())
        goto_joint(arm, planner, q0, 4.0)
        for i, (x, y, z) in enumerate(pts):
            print(f"[{i:2d}/{len(pts)}] ({x:.2f}, {y:.2f}, {z:.2f})", end=" ")
            try:
                if z >= Z_TRAVEL:  # travel move: straight up, over, and skip the blink
                    here = planner.fk_pos(arm.q())
                    move_cartesian(arm, planner, here[0], here[1], Z_TRAVEL, yaw, tilt, 2.0)
                    move_cartesian(arm, planner, x, y, Z_TRAVEL, yaw, tilt, 3.0)
                    print("travel")
                    continue
                move_cartesian(arm, planner, x, y, z, yaw, tilt, 2.5)
                p = settle(arm, planner, x, y, z, yaw, tilt)
            except RuntimeError as e:
                print("skip:", e)
                continue
            time.sleep(0.6)
            fa = {k: c.latest() for k, c in cams.items()}
            time.sleep(0.35)
            fa2 = {k: c.latest() for k, c in cams.items()}
            arm.set_grip(0.0, 1.0)
            time.sleep(0.4)
            fb = {k: c.latest() for k, c in cams.items()}
            arm.set_grip(GRIP_OPEN, 1.0)
            time.sleep(0.4)
            fc = {k: c.latest() for k, c in cams.items()}
            rec = {"i": i, "target": [x, y, z], "fk": [float(v) for v in p], "px": {}}
            for k in cams:
                for tag, f in (("a", fa[k]), ("a2", fa2[k]), ("b", fb[k]), ("c", fc[k])):
                    cv2.imwrite(str(SWEEP / f"{i:02d}_{k}_{tag}.png"), f)
                uv, _ = wedge_tip(fa[k], fb[k], fc[k], fa2[k])
                rec["px"][k] = uv
            print(f"fk {np.round(p, 3)}  " + "  ".join(f"{k} {np.round(v, 0).tolist() if v else None}"
                                                        for k, v in rec["px"].items()))
            records.append(rec)
        park(arm, planner, yaw, tilt)
    except BaseException:
        park(arm, planner, yaw, tilt)  # an error or Ctrl-C mid-sweep: never cut the torque with the arm out
        raise
    finally:
        (SWEEP / "sweep.json").write_text(json.dumps(records, indent=1))
        print(f"saved {len(records)} points -> {SWEEP / 'sweep.json'}")
        arm.close()
        for c in cams.values():
            c.close()


def park(arm, planner, yaw: float, tilt: float) -> None:
    """Back to REST from anywhere in the sweep. Straight up first; where the wrist cannot stay pointing down
    that high (far out, e.g. 38 cm at 20 cm up), go home through READY in joint space instead."""
    from dance import READY
    from pick_bottle import REST, Z_TRAVEL, goto_joint, move_cartesian

    print("returning to rest")
    try:
        if yaw is None:
            raise RuntimeError("the arm never left rest")
        here = planner.fk_pos(arm.q())
        move_cartesian(arm, planner, here[0], here[1], max(here[2], min(Z_TRAVEL, here[2] + 0.05)), yaw, tilt, 1.5)
    except RuntimeError as e:
        print(f"   (no straight lift from here: {e}; going through READY)")
    try:
        goto_joint(arm, planner, READY, 4.0)
        goto_joint(arm, planner, REST, 4.0)
    except RuntimeError as e:
        print(f"!! could not park: {e}")


# ----------------------------------------------------------------------------------------------- solve
def sweep_dirs(args: "Args") -> list[Path]:
    return [HERE / "captures" / d.strip() for d in args.dirs.split(",") if d.strip()]


def load_sweep(redetect: bool = True, dirs: list[Path] | None = None) -> list[dict]:
    recs = []
    for SWEEP in dirs or SWEEPS:
        if not (SWEEP / "sweep.json").exists():
            continue
        for r in json.loads((SWEEP / "sweep.json").read_text()):
            r["dir"] = str(SWEEP)
            if redetect:  # re-run the detector on the saved frames so detector fixes apply without a new sweep
                for k in list(r["px"]):
                    fs = [cv2.imread(str(SWEEP / f"{r['i']:02d}_{k}_{t}.png")) for t in ("a", "b", "c")]
                    a2 = cv2.imread(str(SWEEP / f"{r['i']:02d}_{k}_a2.png"))
                    if all(f is not None for f in fs):
                        r["px"][k], _ = wedge_tip(*fs, a2)
            recs.append(r)
    return recs


def solve_camera(obj: np.ndarray, img: np.ndarray, size: tuple[int, int]) -> dict:
    w, h = size
    f0 = 0.9 * w  # ~60 deg horizontal FOV webcams
    K0 = np.array([[f0, 0, w / 2], [0, f0, h / 2], [0, 0, 1]], dtype=np.float64)
    # one view of a small volume cannot pin down the principal point: fix it at the image centre and
    # solve only f, k1 and the pose
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
             | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [obj.astype(np.float32)], [img.astype(np.float32)], (w, h), K0, None, flags=flags
    )
    R, _ = cv2.Rodrigues(rvecs[0])
    proj, _ = cv2.projectPoints(obj, rvecs[0], tvecs[0], K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
    C = (-R.T @ tvecs[0]).ravel()  # camera centre in the robot frame
    return {
        "K": K.tolist(), "dist": dist.ravel().tolist(), "R": R.tolist(), "t": tvecs[0].ravel().tolist(),
        "size": [w, h], "rms_px": float(rms), "per_point_px": err.tolist(), "center": C.tolist(),
        "fov_deg": float(2 * np.degrees(np.arctan(w / (2 * K[0, 0])))),
    }


def stage_solve(args: Args) -> None:
    recs = load_sweep(dirs=sweep_dirs(args))
    out = {}
    for k in sorted({k for r in recs for k in r["px"]}):
        pts = [(r["fk"], r["px"][k]) for r in recs if r["px"].get(k) is not None]
        if len(pts) < 8:
            print(f"{k}: only {len(pts)} fingertip detections - not enough to solve it (needs 8); left out")
            continue
        obj = np.array([p for p, _ in pts], dtype=np.float64)
        img = np.array([u for _, u in pts], dtype=np.float64)
        f = cv2.imread(str(Path(recs[0]["dir"]) / f"{recs[0]['i']:02d}_{k}_b.png"))
        h, w = f.shape[:2]
        # robust: seed the solution from the sweep with the most trustworthy detections for this camera
        # (cam 1 sits at table level, its clean detections are the low sweep), then admit any point that
        # agrees with that model, and refine.
        # cam 1 looks at the fingers against the robot's own black body, so its automatic detections are
        # unreliable: hand-labelled tips (captures/<sweep>/cam1_labels.json, {"labels": {"<i>": [u, v]}})
        # replace them and seed the solution; automatic points are then admitted only if they agree.
        labelled = np.zeros(len(obj), bool)
        for j, r in enumerate([r for r in recs if r["px"].get(k) is not None]):
            lab = Path(r["dir"]) / f"{k}_labels.json"
            if lab.exists() and str(r["i"]) in json.loads(lab.read_text())["labels"]:
                img[j] = json.loads(lab.read_text())["labels"][str(r["i"])]
                labelled[j] = True
        seed = labelled if labelled.sum() >= 8 else np.ones(len(obj), bool)
        cam = solve_camera(obj[seed], img[seed], (w, h))
        K, dist, R, t = (np.array(cam[x]) for x in ("K", "dist", "R", "t"))
        proj, _ = cv2.projectPoints(obj, cv2.Rodrigues(R)[0], t, K, dist)
        e = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
        good = (e <= max(3 * np.median(e[seed]), 8.0)) | labelled
        if good.sum() >= 8:
            cam = solve_camera(obj[good], img[good], (w, h))
            e2 = np.array(cam["per_point_px"])
            good2 = e2 <= max(3 * np.median(e2), 8.0)
            if 8 <= good2.sum() < len(e2):
                sub = np.where(good)[0][good2]
                good = np.zeros(len(obj), bool); good[sub] = True
                cam = solve_camera(obj[good], img[good], (w, h))
        cam["n_points"] = int(good.sum())
        cam["n_detected"] = len(pts)
        out[k] = cam
        K = np.array(cam["K"])
        print(f"{k}: {cam['n_points']}/{len(recs)} pts (detected {len(pts)}), rms {cam['rms_px']:.2f} px, "
              f"f={K[0,0]:.0f}px ({cam['fov_deg']:.0f} deg HFOV), c=({K[0,2]:.0f},{K[1,2]:.0f}), "
              f"dist {np.round(cam['dist'][:2], 3)}, camera at {np.round(cam['center'], 3)} m")
    CAMERAS.write_text(json.dumps(out, indent=1))
    print(f"saved {CAMERAS}")


def stage_check(args: Args) -> None:
    cams = json.loads(CAMERAS.read_text())
    recs = load_sweep(redetect=False, dirs=sweep_dirs(args))
    for k, cam in cams.items():
        K, dist, R, t = (np.array(cam[x]) for x in ("K", "dist", "R", "t"))
        rvec, _ = cv2.Rodrigues(R)
        print(f"--- {k}")
        for r in recs:
            if r["px"].get(k) is None:
                print(f"  {r['i']:2d} {np.round(r['fk'],3)}  not detected")
                continue
            proj, _ = cv2.projectPoints(np.array([r["fk"]]), rvec, t, K, dist)
            e = np.linalg.norm(proj.ravel() - np.array(r["px"][k]))
            print(f"  {r['i']:2d} {np.round(r['fk'],3)}  px {np.round(r['px'][k])}  proj {np.round(proj.ravel())}  err {e:.1f}")


# ------------------------------------------------------------------------------------- 3D geometry API
class CameraModel:
    def __init__(self, d: dict):
        self.K, self.dist, self.R, self.t = (np.array(d[x], dtype=np.float64) for x in ("K", "dist", "R", "t"))
        self.rvec, _ = cv2.Rodrigues(self.R)
        self.C = -self.R.T @ self.t  # centre in robot frame
        self.size = d["size"]

    def project(self, X: np.ndarray) -> np.ndarray:
        p, _ = cv2.projectPoints(np.asarray(X, dtype=np.float64).reshape(-1, 3), self.rvec, self.t, self.K, self.dist)
        return p.reshape(-1, 2)

    def ray(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """Robot-frame origin and unit direction of the back-projected ray through pixel (u, v)."""
        n = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), self.K, self.dist).ravel()
        d_cam = np.array([n[0], n[1], 1.0])
        d = self.R.T @ d_cam
        return self.C, d / np.linalg.norm(d)

    def hit_plane(self, u: float, v: float, z: float = 0.0) -> np.ndarray:
        o, d = self.ray(u, v)
        s = (z - o[2]) / d[2]
        return o + s * d


def triangulate(c0: CameraModel, uv0, c1: CameraModel, uv1) -> tuple[np.ndarray, float]:
    """Closest point between the two back-projected rays, and the gap between them (m)."""
    o0, d0 = c0.ray(*uv0)
    o1, d1 = c1.ray(*uv1)
    w = o0 - o1
    a, b, c, d, e = d0 @ d0, d0 @ d1, d1 @ d1, d0 @ w, d1 @ w
    den = a * c - b * b
    s = (b * e - c * d) / den
    tt = (a * e - b * d) / den
    p0, p1 = o0 + s * d0, o1 + tt * d1
    return (p0 + p1) / 2, float(np.linalg.norm(p0 - p1))


def load_cameras() -> dict[str, CameraModel]:
    return {k: CameraModel(v) for k, v in json.loads(CAMERAS.read_text()).items()}


def charuco():
    spec = json.loads((HERE / "charuco_board.json").read_text())
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["dictionary"]))
    board = cv2.aruco.CharucoBoard(tuple(spec["squares"]), spec["square_mm"] / 1000, spec["marker_mm"] / 1000, d)
    return board, cv2.aruco.CharucoDetector(board)


def board_corners(img: np.ndarray, board, det) -> tuple[np.ndarray, np.ndarray] | None:
    """ChArUco inner corners seen in img: (board-frame 3D points (N,3), pixels (N,2)), or None if < 6."""
    cc, ci, _, _ = det.detectBoard(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img)
    if ci is None or len(ci) < 6:
        return None
    obj = np.asarray(board.getChessboardCorners())[ci.ravel()]
    return obj.astype(np.float64), cc.reshape(-1, 2).astype(np.float64)


def stage_board(args: Args) -> None:
    """Calibrate cameras from the ChArUco board: the calibrated cameras put the board in the robot frame, then
    each camera in --board-cams is solved from dozens of sharp board corners (intrinsics + pose)."""
    from scene3d import grab

    board, det = charuco()
    raw = json.loads(CAMERAS.read_text())
    known = {k: CameraModel(v) for k, v in raw.items()}
    targets = [k.strip() for k in args.board_cams.split(",") if k.strip()]
    frames = {k: grab(int(k[3:])) for k in sorted(set(known) | set(targets))}
    for k, f in frames.items():
        cv2.imwrite(str(HERE / "captures" / f"board_{k}.png"), f)
    # 1. board -> robot from every calibrated camera that is not being recalibrated
    poses = []
    for k, cam in known.items():
        if k in targets:
            continue
        found = board_corners(frames[k], board, det)
        if found is None:
            print(f"{k}: board not seen")
            continue
        obj, img = found
        ok, rv, tv = cv2.solvePnP(obj, img, cam.K, cam.dist)
        Rbc, _ = cv2.Rodrigues(rv)
        R = cam.R.T @ Rbc
        t = cam.R.T @ (tv.ravel() - cam.t)
        pr, _ = cv2.projectPoints(obj, rv, tv, cam.K, cam.dist)
        e = np.linalg.norm(pr.reshape(-1, 2) - img, axis=1)
        poses.append((k, R, t))
        print(f"{k}: {len(obj)} corners, board fit {np.sqrt(np.mean(e ** 2)):.2f} px, board origin at {np.round(t, 4).tolist()} m")
    if not poses:
        raise SystemExit("no calibrated camera sees the board: it cannot be placed in the robot frame")
    allpts = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    W = [allpts @ R.T + t for _, R, t in poses]
    if len(W) > 1:
        dev = np.linalg.norm(W[0] - W[1], axis=1)
        print(f"the calibrated cameras agree on the board corners to {dev.mean() * 1000:.1f} mm mean, {dev.max() * 1000:.1f} mm max")
    world = np.mean(W, axis=0)  # board corner id -> robot xyz
    print(f"board corner height above the table plane: {world[:, 2].mean() * 1000:+.1f} mm (should be ~0)")
    # 2. each target camera from the board corners
    for k in targets:
        found = board_corners(frames[k], board, det)
        if found is None:
            print(f"{k}: board not seen - not calibrated")
            continue
        obj_b, img = found
        ids = [int(np.argmin(np.linalg.norm(allpts - p, axis=1))) for p in obj_b]
        obj_w = world[ids]
        h, w = frames[k].shape[:2]
        f0 = 1.1 * w
        K0 = np.array([[f0, 0, w / 2], [0, f0, h / 2], [0, 0, 1]], dtype=np.float64)
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
                 | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
        # intrinsics from the planar board in its own frame, then the pose against the robot-frame corners
        rms, K, dist, _, _ = cv2.calibrateCamera([obj_b.astype(np.float32)], [img.astype(np.float32)], (w, h), K0, None, flags=flags)
        ok, rv, tv = cv2.solvePnP(obj_w, img, K, dist)
        R, _ = cv2.Rodrigues(rv)
        pr, _ = cv2.projectPoints(obj_w, rv, tv, K, dist)
        e = np.linalg.norm(pr.reshape(-1, 2) - img, axis=1)
        C = (-R.T @ tv).ravel()
        cam = {"K": K.tolist(), "dist": dist.ravel().tolist(), "R": R.tolist(), "t": tv.ravel().tolist(), "size": [w, h],
               "rms_px": float(np.sqrt(np.mean(e ** 2))), "per_point_px": e.tolist(), "center": C.tolist(),
               "fov_deg": float(2 * np.degrees(np.arctan(w / (2 * K[0, 0])))), "n_points": len(obj_w), "n_detected": len(obj_w),
               "method": f"ChArUco board ({len(obj_w)} corners), board located by {[p[0] for p in poses]}"}
        print(f"{k}: {len(obj_w)} corners, rms {cam['rms_px']:.2f} px, f {K[0, 0]:.0f} px ({cam['fov_deg']:.0f} deg), "
              f"k1 {dist.ravel()[0]:+.3f}, camera at {np.round(C, 3).tolist()} m")
        if cam["rms_px"] > 3.0:
            print(f"   {k}: rms above 3 px - NOT saved")
            continue
        raw[k] = cam
    (HERE / "captures" / "cameras_before_board.json").write_text(CAMERAS.read_text())
    CAMERAS.write_text(json.dumps(raw, indent=1))
    print(f"saved {CAMERAS} ({list(raw)}); previous copy in captures/cameras_before_board.json")


def stage_markers(args: Args) -> None:
    """ArUco markers on the table: the reference camera back-projects every marker corner onto z = 0 (the
    sheet is flat on the table), which puts it in the robot frame whatever the board's layout or scale. Each
    target camera is then calibrated (focal length, k1, pose) from the corners it shares with the reference,
    with a third of the markers held out to check the fit."""
    from scene3d import grab

    raw = json.loads(CAMERAS.read_text())
    ref = CameraModel(raw[args.ref])
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000),
                                  cv2.aruco.DetectorParameters())
    targets = [k.strip() for k in args.board_cams.split(",") if k.strip()]
    frames = {k: grab(int(k[3:])) for k in [args.ref] + targets}
    found = {}
    for k, f in frames.items():
        cv2.imwrite(str(HERE / "captures" / f"markers_{k}.png"), f)
        c, ids, _ = det.detectMarkers(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        found[k] = {} if ids is None else {int(i): cc.reshape(4, 2) for i, cc in zip(ids.ravel(), c)}
        print(f"{k}: {len(found[k])} markers")
    world = {i: np.array([ref.hit_plane(float(u), float(v), 0.0) for u, v in cc]) for i, cc in found[args.ref].items()}
    for k in targets:
        common = sorted(set(world) & set(found[k]))
        if len(common) < 6:
            print(f"{k}: only {len(common)} markers shared with {args.ref} - not calibrated")
            continue
        hold = common[::3]  # every third marker is kept out of the fit
        fit_ids = [i for i in common if i not in hold]
        obj = np.vstack([world[i] for i in fit_ids]).astype(np.float32)
        img = np.vstack([found[k][i] for i in fit_ids]).astype(np.float32)
        h, w = frames[k].shape[:2]
        obj[:, 2] = 0.0  # planar: all on the table
        if k in raw:
            # a bumped camera keeps its lens: only its pose is re-solved (fitting the lens to a patch of
            # markers overfits it - f 1934 px, k1 +0.93 - and the zone then lands 2 cm off)
            K, dist = np.array(raw[k]["K"], dtype=np.float64), np.array(raw[k]["dist"], dtype=np.float64)
            ok, rv, tv, inl = cv2.solvePnPRansac(obj.astype(np.float64), img.astype(np.float64), K, dist,
                                                 reprojectionError=4.0, iterationsCount=5000)
            rv, tv = cv2.solvePnPRefineLM(obj.astype(np.float64)[inl.ravel()], img.astype(np.float64)[inl.ravel()], K, dist, rv, tv)
            rvs, tvs = [rv], [tv]
        else:
            K0 = np.array([[1.1 * w, 0, w / 2], [0, 1.1 * w, h / 2], [0, 0, 1]], dtype=np.float64)
            flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
                     | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
            rms, K, dist, rvs, tvs = cv2.calibrateCamera([obj], [img], (w, h), K0, None, flags=flags)
        R, _ = cv2.Rodrigues(rvs[0])
        tv = tvs[0].ravel()
        def err(ids):
            o = np.vstack([world[i] for i in ids]); o[:, 2] = 0.0
            pr, _ = cv2.projectPoints(o, rvs[0], tvs[0], K, dist)
            return np.linalg.norm(pr.reshape(-1, 2) - np.vstack([found[k][i] for i in ids]), axis=1)
        e_fit, e_hold = err(fit_ids), err(hold)
        C = (-R.T @ tv).ravel()
        print(f"{k}: {len(common)} shared markers ({4 * len(fit_ids)} corners fitted, {4 * len(hold)} held out): "
              f"fit {np.sqrt(np.mean(e_fit ** 2)):.2f} px, held-out {np.sqrt(np.mean(e_hold ** 2)):.2f} px, "
              f"f {K[0, 0]:.0f} px, k1 {dist.ravel()[0]:+.3f}, camera at {np.round(C, 3).tolist()} m")
        if np.sqrt(np.mean(e_hold ** 2)) > 4.0:
            print(f"   {k}: held-out error above 4 px - NOT saved")
            continue
        raw[k] = {"K": K.tolist(), "dist": dist.ravel().tolist(), "R": R.tolist(), "t": tv.tolist(), "size": [w, h],
                  "rms_px": float(np.sqrt(np.mean(e_fit ** 2))), "heldout_px": float(np.sqrt(np.mean(e_hold ** 2))),
                  "per_point_px": e_fit.tolist(), "center": C.tolist(), "fov_deg": float(2 * np.degrees(np.arctan(w / (2 * K[0, 0])))),
                  "n_points": 4 * len(fit_ids), "n_detected": 4 * len(common),
                  "method": f"ArUco markers on the table, placed in the robot frame by {args.ref}"}
    (HERE / "captures" / "cameras_before_markers.json").write_text(CAMERAS.read_text())
    CAMERAS.write_text(json.dumps(raw, indent=1))
    print(f"saved {CAMERAS} ({list(raw)})")


def main(args: Args) -> None:
    if args.markers:
        return stage_markers(args)
    if args.board:
        return stage_board(args)
    if args.sweep:
        stage_sweep(args)
    elif args.solve:
        stage_solve(args)
    elif args.check:
        stage_check(args)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main(tyro.cli(Args))
