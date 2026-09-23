"""Build a 3D scene of the workspace from both calibrated cameras + the arm's joint state.

    python scene3d.py                 # capture both cameras, read the arm pose (CAN, no motion) -> scene.json
    python scene3d.py --no-arm        # use the REST pose instead of reading the arm
    python scene3d.py --frames captures/now_cam0.jpg captures/now_cam1.jpg

Objects: everything that differs from an empty-table background frame. For each blob in cam 0 the table
contact (lowest full-width row) is intersected with the z=0 plane -> footprint centre, the silhouette
width gives the diameter, and the top row gives the height. An orange cap, if present, is triangulated
between the two cameras as an independent check. Robot: MuJoCo meshes posed by FK.
Output scene.json is consumed by scene_viewer.html (three.js).
"""

from __future__ import annotations

import base64
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np
import tyro

from calib3d import CameraModel, load_cameras, triangulate

HERE = Path(__file__).resolve().parent
BACKGROUND = {f"cam{i}": HERE / f"captures/bg_cam{i}.png" for i in range(3)}  # empty table, captured just before the objects were placed
TABLE_ROI_CAM0 = (200, 0, 1180, 720)  # v0, u0, u1, v1: below the base plate, left of the operator


@dataclass
class Args:
    frames: tuple[str, ...] = ()
    no_arm: bool = False
    cam: int = 0
    cam2: int = 1
    out: str = "scene.json"


def grab(index: int) -> np.ndarray:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    t0 = time.time()
    while time.time() - t0 < 1.5:
        cap.read()
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"camera {index} read failed")
    return f


# ------------------------------------------------------------------------------------------- objects
def foreground(img: np.ndarray, bg: np.ndarray, roi=None, thresh: int = 20) -> np.ndarray:
    m = (cv2.absdiff(img, bg).max(axis=2) > thresh).astype(np.uint8) * 255
    win = np.full_like(m, 255)
    if roi:
        v0, u0, u1, v1 = roi
        win[:] = 0
        win[v0:v1, u0:u1] = 255
        m = cv2.bitwise_and(m, win)
    # cast shadows are the table, only darker: same hue, lower value. Drop them or the object's footprint
    # lands on the shadow instead of where the object touches the table.
    # A shadow pixel still looks like wood (table hue ~20, saturation 35-90), just darker than the
    # background there. Objects (white cup S<30, coloured prints, blue/orange plastic) fail the hue/sat test.
    hi, hb = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16), cv2.cvtColor(bg, cv2.COLOR_BGR2HSV).astype(np.int16)
    wood = (hi[..., 0] >= 10) & (hi[..., 0] <= 30) & (hi[..., 1] >= 35) & (hi[..., 1] <= 95)
    shadow = wood & (hi[..., 2] < hb[..., 2])
    m[shadow] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    return m


def orange_centroid(img: np.ndarray) -> tuple[float, float] | None:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (8, 140, 180), (30, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 400]
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    mu = cv2.moments(c)
    return mu["m10"] / mu["m00"], mu["m01"] / mu["m00"]


def object_from_blob(cam: CameraModel, sil: np.ndarray, cap_uv=None) -> dict | None:
    rows = np.where(sil.any(axis=1))[0]
    if len(rows) < 20:
        return None
    widths = np.array([np.ptp(np.where(sil[v] > 0)[0]) for v in rows])
    wmax = np.percentile(widths, 90)
    # lowest row that is still a real slice of the object (the front contact point), not a noise sliver;
    # shadows are already removed so a low threshold is safe, and tapered cups narrow towards the base
    body = rows[widths >= 0.25 * wmax]
    v_b = int(body.max())
    cols = np.where(sil[v_b] > 0)[0]
    u_b = (cols.min() + cols.max()) / 2
    front = cam.hit_plane(u_b, v_b, 0.0)
    # diameter: silhouette width a little above the (rounded) bottom, mapped on the contact row
    v_w = max(rows.min(), v_b - 20)
    cw = np.where(sil[v_w] > 0)[0]
    L, R = cam.hit_plane(cw.min(), v_b, 0.0), cam.hit_plane(cw.max(), v_b, 0.0)
    diam = float(np.linalg.norm(R - L))
    tc = cam.C[:2] - front[:2]
    tc /= np.linalg.norm(tc)
    axis = front[:2] - diam / 2 * tc
    # height: the top-most silhouette row on the axis' image column, intersected with the vertical axis line
    v_t = int(rows.min())
    u_t = float(np.mean(np.where(sil[v_t] > 0)[0]))
    o, d = cam.ray(u_t, v_t)
    # closest point on the ray to the vertical line x=axis: solve for the ray parameter minimising xy distance
    s = -((o[:2] - axis) @ d[:2]) / (d[:2] @ d[:2])
    top = o + s * d
    return {
        "axis": [float(axis[0]), float(axis[1])],
        "diameter": diam,
        "height": float(max(top[2], 0.0)),
        "front_contact": [float(front[0]), float(front[1])],
        "px_bottom": [float(u_b), float(v_b)],
        "px_top": [u_t, float(v_t)],
    }


def detect_objects(frames: dict[str, np.ndarray], cams: dict[str, CameraModel]) -> list[dict]:
    c0 = cams["cam0"]
    bg = cv2.imread(str(BACKGROUND["cam0"]))
    fg = foreground(frames["cam0"], bg, TABLE_ROI_CAM0)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objs = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < 4000:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if bx <= 2 or bx + bw >= fg.shape[1] - 2:  # cut by the frame edge (people at the sides): unusable
            continue
        sil = np.zeros_like(fg)
        cv2.drawContours(sil, [c], -1, 255, -1)
        o = object_from_blob(c0, sil)
        if o is None or not (0.12 < np.hypot(*o["axis"]) < 0.65) or o["diameter"] > 0.25:
            continue
        if o["height"] < 0.04 or o["diameter"] < 0.02:  # cable shifts, glare: not something to pick
            continue
        o["name"] = f"object{len(objs)}"
        o["color"] = [int(v) for v in cv2.mean(frames["cam0"], sil)[:3][::-1]]  # RGB
        objs.append(o)
    # orange cap: triangulate between the cameras as a cross-check on the first object that has one
    o0, o1 = orange_centroid(frames["cam0"]), orange_centroid(frames.get("cam1", frames["cam0"]))
    if o0 and o1 and "cam1" in cams:
        X, gap = triangulate(cams["cam0"], o0, cams["cam1"], o1)
        for o in objs:
            if np.hypot(X[0] - o["axis"][0], X[1] - o["axis"][1]) < 0.08:
                o["name"] = "wash bottle (orange cap)"
                o["cap_xyz"] = [float(v) for v in X]
                o["cap_ray_gap"] = gap
                o["cap_px"] = {"cam0": list(o0), "cam1": list(o1)}
                break
    return objs


# --------------------------------------------------------------------------------------------- robot
def robot_meshes(q6: np.ndarray) -> list[dict]:
    from pick_bottle import Planner

    p = Planner()
    m, d = p.model, p.data
    d.qpos[:] = 0
    d.qpos[:6] = q6
    mujoco.mj_kinematics(m, d)
    out = []
    for gi in range(m.ngeom):
        mid = m.geom_dataid[gi]
        v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        f0, nf = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        verts = m.mesh_vert[v0 : v0 + nv]
        faces = m.mesh_face[f0 : f0 + nf]
        R = d.geom_xmat[gi].reshape(3, 3)
        w = verts @ R.T + d.geom_xpos[gi]
        out.append({
            "body": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gi]),
            "vertices": np.round(w, 4).ravel().tolist(),
            "faces": faces.ravel().tolist(),
        })
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    return out, [float(v) for v in d.site_xpos[sid]]


def read_arm_q() -> np.ndarray:
    import yam_mac  # noqa: F401
    from pick_bottle import Arm

    arm = Arm(False, "can0", 100.0)
    try:
        return arm.q()
    finally:
        arm.close()


# ---------------------------------------------------------------------------------------------- main
def camera_entry(k: str, cam: CameraModel, img: np.ndarray, raw: dict) -> dict:
    small = cv2.resize(img, (640, 360))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    w, h = cam.size
    corners = [cam.ray(u, v)[1].tolist() for (u, v) in ((0, 0), (w, 0), (w, h), (0, h))]
    return {
        "name": k, "K": cam.K.tolist(), "dist": cam.dist.tolist(), "R": cam.R.tolist(), "t": cam.t.tolist(),
        "center": cam.C.tolist(), "size": [w, h], "corner_dirs": corners,
        "fov_deg": raw["fov_deg"], "rms_px": raw["rms_px"], "n_points": raw["n_points"],
        "image_jpeg_b64": base64.b64encode(buf.tobytes()).decode(),
    }


def main(args: Args) -> None:
    cams = load_cameras()
    raw = json.loads((HERE / "cameras.json").read_text())
    if args.frames:
        frames = {"cam0": cv2.imread(args.frames[0]), "cam1": cv2.imread(args.frames[1])}
    else:
        frames = {"cam0": grab(args.cam), "cam1": grab(args.cam2)}
        cv2.imwrite(str(HERE / "captures/scene_cam0.jpg"), frames["cam0"])
        cv2.imwrite(str(HERE / "captures/scene_cam1.jpg"), frames["cam1"])
    objs = detect_objects(frames, cams)
    for o in objs:
        print(f"{o['name']}: axis {np.round(o['axis'], 3)}  diameter {o['diameter']*100:.1f} cm  height {o['height']*100:.1f} cm"
              + (f"  cap (triangulated) {np.round(o['cap_xyz'], 3)} ray gap {o['cap_ray_gap']*100:.1f} cm" if "cap_xyz" in o else ""))
    q = np.zeros(6) if args.no_arm else read_arm_q()
    meshes, tcp = robot_meshes(q)
    scene = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frame": "robot base: x forward, y left, z up, table z=0 (m)",
        "cameras": [camera_entry(k, cams[k], frames[k], raw[k]) for k in ("cam0", "cam1")],
        "objects": objs,
        "robot": {"q": [float(v) for v in q], "tcp": tcp, "meshes": meshes},
        "calibration_points": [r["fk"] for r in json.loads((HERE / "captures/sweep/sweep.json").read_text())]
        + [r["fk"] for r in json.loads((HERE / "captures/sweep_low/sweep.json").read_text())],
    }
    Path(args.out).write_text(json.dumps(scene))
    print(f"wrote {args.out} ({Path(args.out).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main(tyro.cli(Args))
