"""The workspace map, live: a three.js page whose arm, trajectory and objects follow a running pick.

    python live_map.py                  # scene.json + the arm model -> workspace_map.html (publish it as the artifact)

The page embeds everything that does not move - camera frusta and images (from scene.json), the calibration
sweep and every arm mesh in its own geom frame - and subscribes to ONE db document, `live/state`. While
`pick_place.py --run` is going, `LiveState` rewrites captures/live/state.json several times a second:

    status   planning | running | done | aborted     phase / steps / step   the waypoint being executed
    q        7 joint values (gripper last)             poses                  one [px py pz r00..r22] per mesh
    path     the planned fingertip path                trail                  where the fingertips actually went
    pick / place / mode / reason / objects             updated                epoch ms

and that file is copied into the artifact's db while the arm runs (the Claude session does the relay), so
the published map animates. Without db (an old viewer, a downloaded copy) the page shows the embedded pose.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
LIVE = HERE / "captures" / "live"
STATE = LIVE / "state.json"
TEMPLATE = HERE / "live_map_template.html"
PAGE = HERE / "workspace_map.html"
JAW_SLIDE = 0.0475  # m of travel per finger at grip 1.0 (two 47.5 mm slides)


def mesh_geoms(model: mujoco.MjModel) -> list[int]:
    """Every geom that has a mesh, in a fixed order the page and the live poses share."""
    return [gi for gi in range(model.ngeom) if model.geom_dataid[gi] >= 0]


def _finger_adr(model: mujoco.MjModel) -> list[int]:
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ("joint7", "joint8")]
    return [model.jnt_qposadr[j] for j in ids if j >= 0]


def geom_poses(model: mujoco.MjModel, data: mujoco.MjData, q7) -> tuple[list[list[float]], list[float]]:
    """Pose of every mesh geom (position + row-major rotation) and the fingertip site, for joint state q7."""
    q7 = np.asarray(q7, dtype=float)
    data.qpos[:] = 0
    data.qpos[:6] = q7[:6]
    for a in _finger_adr(model):
        data.qpos[a] = JAW_SLIDE * float(np.clip(q7[6] if len(q7) > 6 else 1.0, 0.0, 1.0))
    mujoco.mj_kinematics(model, data)
    poses = [[round(float(v), 5) for v in (*data.geom_xpos[gi], *data.geom_xmat[gi])] for gi in mesh_geoms(model)]
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    return poses, [round(float(v), 4) for v in data.site_xpos[sid]]


class LiveState:
    """Writes captures/live/state.json at `hz` from the arm's measured joints, plus what the executor says
    it is doing. Its own MjData: the planner's belongs to the thread that plans and moves."""

    def __init__(self, model: mujoco.MjModel, get_q7: Callable[[], np.ndarray] | None, hz: float = 5.0):
        self.model, self.data, self.get_q7, self.hz = model, mujoco.MjData(model), get_q7, hz
        self.lock = threading.Lock()
        self.doc: dict = {"status": "planning", "phase": "looking at the table", "steps": [], "step": -1,
                          "trail": [], "path": [], "objects": [], "run": time.strftime("%Y%m%d_%H%M%S"),
                          "batch": os.environ.get("YAM_BATCH", "")}
        self.stop_evt = threading.Event()
        self.thread: threading.Thread | None = None
        self.top_fn: Callable[[], str] | None = None  # -> base64 JPEG of the fused top view
        self.top_every, self._top_t = 2.0, 0.0
        LIVE.mkdir(parents=True, exist_ok=True)

    def set(self, **kw) -> None:
        with self.lock:
            self.doc.update(kw)
        if self.thread is None:  # not sampling the arm yet: write through
            self.write()

    def start(self) -> None:
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self, status: str) -> None:
        self.stop_evt.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.thread = None
        self.set(status=status)

    def _loop(self) -> None:
        while not self.stop_evt.is_set():
            try:
                self.sample()
                if self.top_fn is not None and time.time() - self._top_t >= self.top_every:
                    self._top_t = time.time()
                    top = self.top_fn()
                    with self.lock:
                        self.doc["top"] = top
                self.write()
            except Exception as e:  # never let the map take the run down
                print(f"   (live map: {e})")
            self.stop_evt.wait(1.0 / self.hz)

    def sample(self) -> None:
        if self.get_q7 is None:
            return
        q7 = np.asarray(self.get_q7(), dtype=float)
        poses, tcp = geom_poses(self.model, self.data, q7)
        with self.lock:
            trail = self.doc["trail"]
            if not trail or np.linalg.norm(np.array(tcp) - np.array(trail[-1])) > 0.005:
                trail.append(tcp)
                del trail[:-600]
            self.doc.update(q=[round(float(v), 4) for v in q7], poses=poses, tcp=tcp)

    def write(self) -> None:
        with self.lock:
            self.doc["updated"] = int(time.time() * 1000)
            text = json.dumps(self.doc)
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(text)
        os.replace(tmp, STATE)  # the relay never reads half a file


def build(scene_path: Path = HERE / "scene.json", out: Path = PAGE) -> Path:
    """scene.json (cameras, images, calibration, objects, arm pose) + the arm model -> the live page."""
    from pick_bottle import Planner

    scene = json.loads(Path(scene_path).read_text())
    m = Planner().model
    geoms = []
    for gi in mesh_geoms(m):
        mid = m.geom_dataid[gi]
        v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        f0, nf = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        geoms.append({
            "body": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gi]),
            "vertices": np.round(m.mesh_vert[v0 : v0 + nv], 4).ravel().tolist(),
            "faces": m.mesh_face[f0 : f0 + nf].ravel().tolist(),
        })
    q7 = list(scene["robot"]["q"])[:6] + [1.0]
    poses, tcp = geom_poses(m, mujoco.MjData(m), q7)
    page = {
        "generated": scene["generated"], "frame": scene["frame"], "cameras": scene["cameras"],
        "calibration_points": scene["calibration_points"],
        "robot": {"geoms": geoms},
        "zone": json.loads((HERE / "zone.json").read_text())["polygon"] if (HERE / "zone.json").exists() else [],
        # what the page shows until (or without) a live update
        "top_meta": scene.get("top", {}).get("meta"),
        "initial": {"top": scene.get("top", {}).get("b64"), "status": "idle", "phase": "", "steps": [], "step": -1, "q": q7, "poses": poses, "tcp": tcp,
                    "trail": [], "path": [], "updated": 0,
                    "objects": [{"name": o["name"], "axis": o["axis"], "diameter": o["diameter"],
                                 "height": o["height"], "role": "obstacle"} for o in scene["objects"]]},
    }
    text = json.dumps(page).replace("</", "<\\/")
    Path(out).write_text(TEMPLATE.read_text().replace("/*SCENE_JSON*/", text))
    print(f"wrote {out} ({Path(out).stat().st_size / 1e6:.1f} MB, {len(geoms)} arm meshes)")
    return Path(out)


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "scene.json")
