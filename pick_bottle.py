"""Pick up the blue bottle with the YAM arm using the overhead camera.

Stages (run in order; each one is explicit so the arm never moves unexpectedly):

    python pick_bottle.py --detect            # camera only: find the bottle, write captures/detect.jpg
    python pick_bottle.py --calibrate         # ARM MOVES: visit 6 points above the table, fit image<->table
                                              #   homography from gripper open/close diffs -> calib.json
    python pick_bottle.py --plan              # camera only: detect, map to robot xy, solve IK; no arm connection
    python pick_bottle.py --pick              # ARM MOVES: detect bottle, map to robot xy, grasp, lift, set down
    python pick_bottle.py --pick --sim        # same, MuJoCo only (uses last captures/ image)
    python pick_bottle.py --pick --record datasets/yam_pick_bottle   # also record a LeRobot v2.1 dataset
                                              #   (both cameras + joint state/action at 30 fps)

Frames: robot base x forward, z up, table ~ z=0. Gripper index 6 in joint vector, 1=open 0=closed.
grasp_site (the IK target) is the fingertip distal facet, so "z" below is fingertip height above the table.

The camera is low and oblique (~30 deg elevation), so a table-plane homography is only valid between
features at the SAME height: calibration uses the fingertips at Z_CAL, and the pick uses the centre of
the bottle's silver cap top, which sits at ~Z_CAL on the bottle axis.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mink
import mujoco
import numpy as np
import tyro

import yam_mac  # noqa: F401  (macOS CAN shim; must precede i2rt imports)
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
from lerobot_record import EpisodeRecorder, LiveCamera

HERE = Path(__file__).resolve().parent
CAPTURES = HERE / "captures"
CALIB = HERE / "calib.json"

REST = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
Z_TRAVEL = 0.20  # fingertip height when moving around (bottle cap top is ~0.09)
Z_CAL = 0.09  # calibration plane = height of the bottle's cap top
Z_GRASP = 0.03  # fingertip height when closing: pads then cover the 7 cm body from 3 cm up
Z_SETTLE_TOL = 0.003  # closed-loop settle: re-command until measured fingertip z is within this of target
SERVO_OFFSET = 0.08  # m in front of the bottle (towards the camera) where the gripper blinks for the visual check
SERVO_TOL = 0.004  # m, stop iterating the visual correction below this
SIDE_TILT = 1.05  # rad (60 deg from vertical) for --grasp side: fingers reach the body from behind/above
SIDE_APPROACH = 0.10  # m, pre-grasp distance back along the gripper axis for --grasp side
GRIP_OPEN = 1.0
GRIP_CLOSED_ON_BOTTLE = 0.42  # ~4 cm of a 9.6 cm stroke: firm on a ~5.7 cm bottle
MAX_CAL_ERR = 0.015  # m, refuse a calibration whose worst reprojection error exceeds this
# all inside cam 0's field of view: it sees the table from the robot's front-right, roughly y in [-0.3, 0.05]
CAL_POINTS = [(0.30, 0.00), (0.35, -0.15), (0.25, -0.25), (0.45, -0.05), (0.40, -0.20), (0.22, -0.10)]


@dataclass
class Args:
    detect: bool = False
    calibrate: bool = False
    plan: bool = False
    pick: bool = False
    sim: bool = False
    cam: int = 0
    """OpenCV index of the overhead camera (C270)."""
    cam2: int = 1
    """OpenCV index of the second (side) camera, recorded only."""
    channel: str = "can0"
    hz: float = 100.0
    record: str | None = None
    """Directory to write a LeRobot v2.1 dataset of the pick into (both cameras + state/action)."""
    fps: int = 30
    task: str = "pick up the blue bottle"
    target: str | None = None
    """Skip detection: robot-frame xy of the object's axis, e.g. "0.29,-0.22" (from scene3d.py)."""
    z_grasp: float | None = None
    """Fingertip height when closing (default Z_GRASP)."""
    contact_close: bool = False
    """Close the gripper until it stalls on the object (+ a small squeeze) instead of to a fixed width."""
    squeeze: float = 0.08
    """Extra closing after contact, in gripper units (0.08 ~ 8 mm). Use ~0.03 for rigid objects like glass."""
    grasp: str = "top"
    """'top': fingers straight down over the cap. 'side': wrist tilted 60 deg, fingers clamp the body from behind."""
    servo: bool = True
    """Blink the gripper at the calibration height next to the bottle and correct the arm's FK error before grasping."""


# ----------------------------------------------------------------------------------------------- camera
class Camera:
    def __init__(self, index: int):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera {index} did not open")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        t0 = time.time()
        while time.time() - t0 < 2.0:  # exposure settle
            self.cap.read()

    def frame(self, n_avg: int = 3) -> np.ndarray:
        for _ in range(4):  # flush buffered stale frames
            self.cap.read()
        acc = None
        for _ in range(n_avg):
            ok, f = self.cap.read()
            if not ok:
                raise RuntimeError("camera read failed")
            acc = f.astype(np.float32) if acc is None else acc + f
        return (acc / n_avg).astype(np.uint8)

    def close(self) -> None:
        self.cap.release()


def detect_bottle(img: np.ndarray) -> tuple[tuple[int, int], tuple[int, int, int, int], np.ndarray]:
    """Return (cap-top centre (u,v), body bbox, annotated image) of the biggest blue blob.

    The silver cap sits directly above the blue body. Its top disc is an ellipse whose widest row is
    its centre row, so: take the silver blob above the body and find the first row (from the top)
    where its width reaches the maximum."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (95, 120, 120), (125, 255, 255))  # bright light-blue plastic (excludes dark blue-grey metal)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("no blue bottle found")
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 1500:
        raise RuntimeError(f"blue blob too small ({cv2.contourArea(c):.0f} px)")
    x, y, w, h = cv2.boundingRect(c)
    # silver cap: its top disc reflects the ceiling and reads neutral-to-bluish (R <= B), while the wood
    # table, even where it is bright/unsaturated glare, always has R > B. Search the body's own column
    # band just above its top edge.
    b_, g_, r_ = [img[..., i].astype(np.int16) for i in range(3)]
    grey = ((r_ - b_) < 2) & (hsv[..., 2] > 50)
    silver = grey.astype(np.uint8) * 255
    win = np.zeros_like(silver)
    win[max(0, y - int(0.9 * w)) : y + int(0.2 * h), x + w // 10 : x + w - w // 10] = 255
    silver = cv2.bitwise_and(silver, win)
    silver = cv2.morphologyEx(silver, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    silver = cv2.morphologyEx(silver, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    scnts, _ = cv2.findContours(silver, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # the cap touches the body's top edge: its bottom row must reach down to about y
    scnts = [sc for sc in scnts if cv2.contourArea(sc) > 0.04 * w * w and cv2.boundingRect(sc)[1] + cv2.boundingRect(sc)[3] >= y - 5]
    if not scnts:
        raise RuntimeError("blue body found but no silver cap above it")
    sc = max(scnts, key=cv2.contourArea)
    cap_mask = np.zeros_like(silver)
    cv2.drawContours(cap_mask, [sc], -1, 255, -1)
    rows = np.where(cap_mask.any(axis=1))[0]
    widths = cap_mask[rows].sum(axis=1) // 255
    v_top = int(rows[0])
    cap_w = float(np.percentile(widths, 75))  # disc diameter in px; robust to reflections that widen a few rows
    # the top disc is an ellipse with vertical semi-axis (cap_w/2)*sin(elevation); the camera sits at
    # ~30 deg elevation, so its centre is cap_w/4 below the top edge.
    v = v_top + int(round(0.25 * cap_w))
    disc = cap_mask.copy()
    disc[v_top + int(0.5 * cap_w) :] = 0  # only the disc rows, not the cylindrical side below
    mu = cv2.moments(disc, binaryImage=True)
    u = int(mu["m10"] / mu["m00"])
    out = img.copy()
    cv2.drawContours(out, [c], -1, (0, 255, 0), 2)
    cv2.drawContours(out, [sc], -1, (255, 0, 255), 2)
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 1)
    cv2.drawMarker(out, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
    return (u, v), (x, y, w, h), out


# cam 0 region that is table (rows below the arm's base plate, columns left of the operator): motion
# outside it (people, hands, the arm's own upper links) must not be mistaken for the gripper fingers.
TABLE_ROI = (210, 0, 1150, 720)  # (v_min, u_min, u_max, v_max) in a 1280x720 frame


def detect_motion_blob(
    a: np.ndarray, b: np.ndarray, min_area: float = 150.0, roi: tuple[int, int, int, int] = TABLE_ROI
) -> tuple[tuple[int, int] | None, np.ndarray]:
    """Fingertip pixel (u,v) from the gripper open->closed diff, plus the motion mask for debugging.
    The tips are the lowest moving pixels in the image (the camera looks down at the table)."""
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    d = cv2.GaussianBlur(cv2.absdiff(ga, gb), (5, 5), 0)
    _, m = cv2.threshold(d, 25, 255, cv2.THRESH_BINARY)
    # the fingers are black plastic. Keep only pixels that BECAME dark (open -> closed): that is the
    # closed wedge, whose lowest point is the centred fingertip. Pixels that were dark and cleared are
    # the two open-finger positions, and the fingers' shadow never gets this dark.
    became_dark = ((gb < 70) & (ga >= 70)).astype(np.uint8) * 255
    m = cv2.bitwise_and(m, cv2.dilate(became_dark, np.ones((3, 3), np.uint8)))
    v0, u0, u1, v1 = roi
    win = np.zeros_like(m)
    win[v0:v1, u0:u1] = 255
    m = cv2.bitwise_and(m, win)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.dilate(m, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if not cnts:
        return None, m
    # the wedge is by far the biggest blob; slivers from the arm's cable twitching are not
    big = max(cv2.contourArea(c) for c in cnts)
    keep = np.zeros_like(m)
    cv2.drawContours(keep, [c for c in cnts if cv2.contourArea(c) >= 0.3 * big], -1, 255, -1)
    vs, us = np.where(keep > 0)
    v_tip = vs.max()
    sel = vs >= v_tip - 6  # the bottom few rows: both fingertips
    return (int(us[sel].mean()), int(vs[sel].mean())), keep


# ------------------------------------------------------------------------------------------- kinematics
class Planner:
    def __init__(self) -> None:
        self.xml = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)
        self.kin = Kinematics(self.xml, "grasp_site")
        self.model = self.kin._configuration.model
        self.data = mujoco.MjData(self.model)
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.base_geom_z = self._min_geom_z(np.zeros(6))  # the base's own lowest geom (never moves)

    @staticmethod
    def topdown(x: float, y: float, z: float, yaw: float, tilt: float = 0.0) -> np.ndarray:
        """Gripper pointing down, optionally tilted `tilt` rad outward (away from the base) so the
        wrist-pitch limit can be respected close to the base; `yaw` spins the fingers about that axis."""
        az = np.arctan2(y, x)
        r = np.array([np.cos(az), np.sin(az), 0.0])
        t = np.array([-np.sin(az), np.cos(az), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        zg = np.sin(tilt) * r - np.cos(tilt) * up
        xg0 = np.cos(tilt) * r + np.sin(tilt) * up
        xg = np.cos(yaw) * xg0 + np.sin(yaw) * t
        yg = np.cross(zg, xg)
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = xg, yg, zg, [x, y, z]
        return T

    def ik(
        self, x: float, y: float, z: float, q_init: np.ndarray, yaw: float | None = None, tilt: float | None = None
    ) -> tuple[np.ndarray, float, float]:
        # Seeds: the current pose, then a canonical "wrist down, base pointed at the target" pose.
        # From the folded rest pose alone the solver gets stuck away from the wrist-down solutions.
        seeds = [np.array(q_init[:6]), np.array([np.arctan2(y, x), 1.2, 1.2, -1.5, 0.0, 0.0])]
        yaws = [yaw] if yaw is not None else [np.pi, 0.0] + list(np.linspace(-np.pi, np.pi, 8, endpoint=False))
        tilts = [tilt] if tilt is not None else [0.0, 0.25, 0.45, 0.6]
        for tl in tilts:
            for seed in seeds:
                q0 = np.zeros(self.model.nq)
                q0[:6] = seed
                for yw in yaws:
                    ok, q = self.kin.ik(
                        self.topdown(x, y, z, yw, tl), "grasp_site", init_q=q0, limits=self.limits,
                        max_iters=400, pos_threshold=2e-3, ori_threshold=3e-2,
                    )
                    if ok:
                        return np.array(q[:6]), yw, tl
        raise RuntimeError(f"IK failed for ({x:.3f}, {y:.3f}, {z:.3f})")

    def fk_pos(self, q6: np.ndarray) -> np.ndarray:
        q = np.zeros(self.model.nq)
        q[:6] = q6
        return self.kin.fk(q)[:3, 3]

    def _min_geom_z(self, q6: np.ndarray) -> float:
        self.data.qpos[:6] = q6
        mujoco.mj_kinematics(self.model, self.data)
        return float(self.data.geom_xpos[:, 2].min())

    def path_is_safe(self, q_from: np.ndarray, q_to: np.ndarray, z_floor: float = 0.02) -> bool:
        """Every geom stays above z_floor (except the fixed base) along the straight joint-space path."""
        for a in np.linspace(0, 1, 30):
            q = (1 - a) * q_from + a * q_to
            self.data.qpos[:6] = q
            mujoco.mj_kinematics(self.model, self.data)
            zs = self.data.geom_xpos[:, 2]
            moving = zs[zs > self.base_geom_z + 1e-6] if len(zs) else zs
            if len(moving) and moving.min() < z_floor:
                return False
        return True


# ------------------------------------------------------------------------------------------------ robot
class Arm:
    def __init__(self, sim: bool, channel: str, hz: float):
        self.robot = get_yam_robot(channel=channel, arm_type=ArmType.YAM, gripper_type=GripperType.LINEAR_4310, sim=sim)
        self.hz = hz
        self.grip = GRIP_OPEN
        self.last_cmd = np.array(self.robot.get_joint_pos()[:7], dtype=np.float32)

    def q(self) -> np.ndarray:
        return np.array(self.robot.get_joint_pos()[:6])

    def state7(self) -> np.ndarray:
        return np.array(self.robot.get_joint_pos()[:7], dtype=np.float32)

    def alive(self) -> bool:
        chain = getattr(self.robot, "motor_chain", None)
        return chain is None or getattr(chain, "running", True)

    def _cmd(self, q6: np.ndarray, grip: float | None = None) -> None:
        if not self.alive():
            raise RuntimeError("control loop died")
        g = self.grip if grip is None else grip
        cmd = np.concatenate([q6, [g]])
        self.last_cmd = cmd.astype(np.float32)
        self.robot.command_joint_pos(cmd)

    def glide(self, q_to: np.ndarray, seconds: float) -> None:
        q_from = self.q()
        steps = max(1, int(seconds * self.hz))
        for i in range(steps + 1):
            a = i / steps
            a = 0.5 - 0.5 * np.cos(np.pi * a)  # ease in/out
            self._cmd((1 - a) * q_from + a * q_to)
            time.sleep(seconds / steps)

    def set_grip(self, g: float, seconds: float = 1.5) -> None:
        q6 = np.array(self.last_cmd[:6], dtype=float)  # hold the commanded pose, not the sagging measured one
        g0 = self.grip
        steps = max(1, int(seconds * self.hz))
        for i in range(steps + 1):
            a = i / steps
            self.grip = (1 - a) * g0 + a * g
            self._cmd(q6)
            time.sleep(seconds / steps)

    def close(self) -> None:
        self.robot.close()


def move_cartesian(
    arm: Arm, planner: Planner, x: float, y: float, z: float, yaw: float, tilt: float, seconds: float, step: float = 0.02
) -> np.ndarray:
    """Straight-ish line in Cartesian space: IK on waypoints every `step` metres, joint glide between."""
    q = arm.q()
    p0 = planner.fk_pos(q)
    p1 = np.array([x, y, z])
    n = max(1, int(np.linalg.norm(p1 - p0) / step))
    for i in range(1, n + 1):
        p = p0 + (p1 - p0) * i / n
        q_next, _, _ = planner.ik(p[0], p[1], p[2], q, yaw, tilt)
        if not planner.path_is_safe(q, q_next):
            raise RuntimeError(f"unsafe segment towards {p}")
        arm.glide(q_next, seconds / n)
        q = q_next
    return q


def settle(arm: Arm, planner: Planner, x: float, y: float, z: float, yaw: float, tilt: float, tries: int = 4) -> np.ndarray:
    """The real arm sags a couple of cm under position control at full reach. Measure the fingertip
    position, re-command the target offset by the error, repeat until within Z_SETTLE_TOL."""
    for _ in range(tries):
        p = planner.fk_pos(arm.q())
        err = np.array([x, y, z]) - p
        if np.abs(err).max() <= Z_SETTLE_TOL:
            break
        cmd = np.array([x, y, z]) + err
        q_next, _, _ = planner.ik(cmd[0], cmd[1], cmd[2], arm.q(), yaw, tilt)
        arm.glide(q_next, 0.8)
        time.sleep(0.3)
    return planner.fk_pos(arm.q())


def goto_joint(arm: Arm, planner: Planner, q_to: np.ndarray, seconds: float) -> None:
    if not planner.path_is_safe(arm.q(), q_to):
        raise RuntimeError("unsafe joint path")
    arm.glide(q_to, seconds)


# ----------------------------------------------------------------------------------------------- stages
def stage_detect(args: Args, cam: Camera | None = None) -> tuple[int, int]:
    CAPTURES.mkdir(exist_ok=True)
    if cam is None:
        img = cv2.imread(str(CAPTURES / "cam0.jpg")) if args.sim else Camera(args.cam).frame()
    elif isinstance(cam, LiveCamera):
        img = cam.latest()
    else:
        img = cam.frame()
    cv2.imwrite(str(CAPTURES / "detect_raw.jpg"), img)
    (u, v), bbox, out = detect_bottle(img)
    cv2.imwrite(str(CAPTURES / "detect.jpg"), out)
    print(f"bottle cap-top centre: pixel ({u}, {v}), body bbox {bbox}  -> captures/detect.jpg")
    return u, v


def stage_calibrate(args: Args) -> None:
    CAPTURES.mkdir(exist_ok=True)
    cam = Camera(args.cam)
    planner = Planner()
    arm = Arm(args.sim, args.channel, args.hz)
    pts_robot, pts_img = [], []
    try:
        print("start pos:", np.round(arm.q(), 2))
        arm.set_grip(GRIP_OPEN, 1.0)
        # first travel pose above the first point
        q_travel, yaw, tilt = planner.ik(CAL_POINTS[0][0], CAL_POINTS[0][1], Z_TRAVEL, arm.q())
        goto_joint(arm, planner, q_travel, 4.0)
        for i, (x, y) in enumerate(CAL_POINTS):
            print(f"[{i}] point ({x:.2f}, {y:.2f})")
            try:
                q_here, yaw, tilt = planner.ik(x, y, Z_TRAVEL, arm.q())
                goto_joint(arm, planner, q_here, 3.0)
                move_cartesian(arm, planner, x, y, Z_CAL, yaw, tilt, 2.5)
            except RuntimeError as e:
                print("   skip:", e)
                continue
            p = settle(arm, planner, x, y, Z_CAL, yaw, tilt)
            time.sleep(0.8)
            a = cam.frame()
            arm.set_grip(0.0, 1.2)
            time.sleep(0.5)
            b = cam.frame()
            arm.set_grip(GRIP_OPEN, 1.2)
            uv, mask = detect_motion_blob(a, b)
            print(f"   fingertips actually at {np.round(p, 3)}; image tip: {uv}")
            dbg = b.copy()
            cv2.imwrite(str(CAPTURES / f"cal_{i}_a.png"), a)
            cv2.imwrite(str(CAPTURES / f"cal_{i}_b.png"), b)
            cv2.imwrite(str(CAPTURES / f"cal_{i}_mask.png"), mask)
            if uv and abs(p[2] - Z_CAL) <= 0.01:
                cv2.drawMarker(dbg, uv, (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
                pts_robot.append([float(p[0]), float(p[1])])
                pts_img.append([float(uv[0]), float(uv[1])])
            else:
                print("   discarded (no tip found or not on the calibration plane)")
            cv2.imwrite(str(CAPTURES / f"cal_{i}.jpg"), dbg)
            move_cartesian(arm, planner, x, y, Z_TRAVEL, yaw, tilt, 2.0)
        print("returning to rest")
        goto_joint(arm, planner, REST, 5.0)
    finally:
        arm.close()
        cam.close()
    if len(pts_robot) < 4:
        raise RuntimeError(f"only {len(pts_robot)} calibration points detected; need >= 4")
    H, inl = cv2.findHomography(np.array(pts_img), np.array(pts_robot), 0)  # least squares over all points
    err = [float(np.linalg.norm(img_to_robot(H, u, v) - np.array(r))) for (u, v), r in zip(pts_img, pts_robot)]
    print("reprojection error per point (m):", np.round(err, 3))
    if max(err) > MAX_CAL_ERR:
        raise RuntimeError(f"worst reprojection error {max(err):.3f} m > {MAX_CAL_ERR}; calib.json NOT written")
    CALIB.write_text(json.dumps({"H": H.tolist(), "z_plane": Z_CAL, "pts_img": pts_img, "pts_robot": pts_robot}, indent=2))
    print(f"saved {CALIB}")


def img_to_robot(H: np.ndarray, u: float, v: float) -> np.ndarray:
    p = H @ np.array([u, v, 1.0])
    return p[:2] / p[2]


def robot_to_img(H: np.ndarray, x: float, y: float) -> np.ndarray:
    p = np.linalg.inv(H) @ np.array([x, y, 1.0])
    return p[:2] / p[2]


def towards_camera(H: np.ndarray, u: float, v: float) -> np.ndarray:
    """Unit table-plane direction that moves a point down the image, i.e. towards the camera."""
    d = img_to_robot(H, u, v + 40) - img_to_robot(H, u, v)
    return d / (np.linalg.norm(d) + 1e-9)


def blink_and_locate(arm: Arm, cam: LiveCamera, roi=(100, 0, 1150, 720)) -> tuple[tuple[int, int] | None, np.ndarray, np.ndarray]:
    """Close and reopen the gripper; return the closed fingertip pixel and the two frames."""
    time.sleep(0.6)
    a = cam.latest()
    arm.set_grip(0.0, 1.0)
    time.sleep(0.4)
    b = cam.latest()
    arm.set_grip(GRIP_OPEN, 1.0)
    uv, _ = detect_motion_blob(a, b, roi=roi)
    return uv, a, b


def visual_correct_3d(
    arm: Arm, planner: Planner, cam: LiveCamera, target: np.ndarray, yaw: float, tilt: float, z_check: float, max_iter: int = 3,
    cam_key: str = "cam0", offset: float = SERVO_OFFSET, direction: np.ndarray | None = None,
) -> np.ndarray:
    """Same idea as visual_correct, but with cam 0's calibrated model (cameras.json): the fingertip pixel
    is back-projected onto the horizontal plane at the FK height, so the check can happen at any height."""
    from calib3d import load_cameras, wedge_tip

    c0 = load_cameras()[cam_key]  # the camera `cam` is
    d = c0.C[:2] - target if direction is None else np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)  # towards the camera (or the given direction), in the table plane
    desired = target + offset * d  # where the tips should be seen, fixed for the whole loop
    corr = np.zeros(2)  # accumulated (actual - commanded) offset of the real arm
    for it in range(max_iter):
        check = desired - corr
        move_cartesian(arm, planner, check[0], check[1], z_check, yaw, tilt, 2.0)
        settle(arm, planner, check[0], check[1], z_check, yaw, tilt)
        time.sleep(0.6)
        a = cam.latest(); time.sleep(0.3); a2 = cam.latest()
        arm.set_grip(0.0, 1.0); time.sleep(0.4)
        b = cam.latest()
        arm.set_grip(GRIP_OPEN, 1.0); time.sleep(0.4)
        c = cam.latest()
        believed = planner.fk_pos(arm.q())
        # look for the blink only around where the calibrated camera expects the tips: across the whole
        # frame a person walking behind or another robot's LEDs won (a 0.52 m "error", zone batch)
        eu, ev = c0.project(believed[None])[0]
        H, W = a.shape[:2]
        x0, x1 = int(np.clip(eu - 170, 0, W)), int(np.clip(eu + 170, 0, W))
        y0, y1 = int(np.clip(ev - 170, 0, H)), int(np.clip(ev + 170, 0, H))
        uv = None
        if x1 - x0 > 40 and y1 - y0 > 40:
            crop = lambda f: f[y0:y1, x0:x1]
            uv, _ = wedge_tip(crop(a), crop(b), crop(c), crop(a2))
            if uv is not None:
                uv = (uv[0] + x0, uv[1] + y0)
        dbg = b.copy()
        if uv:
            cv2.drawMarker(dbg, (int(uv[0]), int(uv[1])), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
        cv2.imwrite(str(CAPTURES / f"servo_{it}.jpg"), dbg)
        if uv is None:
            raise RuntimeError("visual check: could not see the fingertips blink")
        actual = c0.hit_plane(uv[0], uv[1], believed[2])[:2]
        # closed loop on where the tips ARE vs where they SHOULD be (the intended check point); the
        # arm's own FK offset (actual - believed) is only printed for information
        err = actual - desired
        print(f"   servo3d[{it}] desired {np.round(desired, 3)} actual {np.round(actual, 3)} err {np.round(err, 3)}"
              f"  (FK offset {np.round(actual - believed[:2], 3)})")
        if np.linalg.norm(err) > 0.08:
            raise RuntimeError(f"visual check: implausible {np.linalg.norm(err):.3f} m error, aborting")
        corr += err
        if np.linalg.norm(err) < SERVO_TOL:
            break
    if np.linalg.norm(corr) > 0.045:  # this arm has never needed more than ~3 cm: a bigger one is a misdetection
        raise RuntimeError(f"visual check: implausible {np.linalg.norm(corr) * 100:.1f} cm total correction, aborting")
    cmd = target - corr
    print(f"   commanding {np.round(cmd, 3)} to land on {np.round(target, 3)} (correction {np.round(corr, 3)})")
    return cmd


def close_until_contact(arm: Arm, squeeze: float = 0.08, step: float = 0.015, lag: float = 0.05) -> float:
    """Close slowly; when the measured opening stops following the command (it hit something), stop and
    add a small squeeze. Returns the measured opening. A soft squeeze bottle is not crushed this way."""
    g = arm.grip
    while g > 0.0:
        g = max(0.0, g - step)
        arm.grip = g
        arm._cmd(np.array(arm.last_cmd[:6], dtype=float))
        time.sleep(0.08)
        meas = float(arm.state7()[6])
        if meas - g > lag:
            print(f"   contact at opening {meas:.2f} (cmd {g:.2f})")
            arm.grip = max(0.0, meas - squeeze)
            arm._cmd(np.array(arm.last_cmd[:6], dtype=float))
            time.sleep(0.6)
            return float(arm.state7()[6])
    return float(arm.state7()[6])


def visual_correct(
    arm: Arm, planner: Planner, cam: LiveCamera, H: np.ndarray, target: np.ndarray, yaw: float, tilt: float, max_iter: int = 3
) -> np.ndarray:
    """The real arm's FK disagrees with the camera by a few cm depending on the joint configuration.
    Park the fingertips on the calibration plane SERVO_OFFSET in front of the bottle, blink the gripper,
    and measure where the tips really are; return the xy to COMMAND so the tips land on `target`."""
    d = towards_camera(H, *robot_to_img(H, *target))
    desired = target + SERVO_OFFSET * d
    corr = np.zeros(2)
    for it in range(max_iter):
        check = desired - corr
        move_cartesian(arm, planner, check[0], check[1], Z_CAL, yaw, tilt, 2.0)
        settle(arm, planner, check[0], check[1], Z_CAL, yaw, tilt)
        uv, a, b = blink_and_locate(arm, cam)
        believed = planner.fk_pos(arm.q())
        dbg = b.copy()
        if uv:
            cv2.drawMarker(dbg, uv, (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
        cv2.imwrite(str(CAPTURES / f"servo_{it}.jpg"), dbg)
        if uv is None:
            raise RuntimeError("visual check: could not see the fingertips blink")
        actual = img_to_robot(H, *uv)
        err = actual - desired  # where the tips really are minus where they should be
        print(f"   servo[{it}] desired {np.round(desired, 3)} actual {np.round(actual, 3)} err {np.round(err, 3)}"
              f"  (FK offset {np.round(actual - believed[:2], 3)})")
        if np.linalg.norm(err) > 0.08:
            raise RuntimeError(f"visual check: implausible {np.linalg.norm(err):.3f} m error, aborting")
        corr += err
        if np.linalg.norm(err) < SERVO_TOL:
            break
    cmd = target - corr
    print(f"   commanding {np.round(cmd, 3)} to land on {np.round(target, 3)} (correction {np.round(corr, 3)})")
    return cmd


def plan_target(args: Args, cam: Camera | None) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Detect the bottle, map its cap centre to robot xy, and solve the approach IK from REST."""
    calib = json.loads(CALIB.read_text())
    H = np.array(calib["H"])
    if args.target:
        target = np.array([float(v) for v in args.target.split(",")])
        u, v = (int(round(x)) for x in robot_to_img(H, *target))
        print(f"object axis given: robot xy {np.round(target, 3)}")
    else:
        u, v = stage_detect(args, cam)
        target = img_to_robot(H, u, v)  # cap top is on the bottle axis at ~Z_CAL, the calibration plane
        print(f"bottle axis at robot xy {np.round(target, 3)}")
    r = np.hypot(*target)
    if not (0.18 <= r <= 0.55):
        raise RuntimeError(f"target {np.round(target, 3)} is {r:.2f} m from base: outside safe reach")
    planner = Planner()
    if args.grasp == "side":
        q_above, yaw, tilt = planner.ik(target[0], target[1], Z_TRAVEL, REST, yaw=np.pi, tilt=SIDE_TILT)
        pre = np.array([*target, Z_GRASP]) - SIDE_APPROACH * planner.topdown(target[0], target[1], Z_GRASP, yaw, tilt)[:3, 2]
        planner.ik(pre[0], pre[1], pre[2], q_above, yaw, tilt)
    else:
        q_above, yaw, tilt = planner.ik(target[0], target[1], Z_TRAVEL, REST)
    planner.ik(target[0], target[1], Z_GRASP, q_above, yaw, tilt)  # make sure the grasp pose is solvable
    d = towards_camera(H, u, v)
    chk = target + SERVO_OFFSET * d
    planner.ik(chk[0], chk[1], Z_CAL, q_above, yaw, tilt)  # ... and the visual-check pose
    print(f"grasp '{args.grasp}': yaw {yaw:.2f} rad, tilt {np.degrees(tilt):.0f} deg; check point {np.round(chk, 3)}; above-pose q {np.round(q_above, 2)}")
    return target, q_above, yaw, tilt


def stage_plan(args: Args) -> None:
    cam = None if args.sim else Camera(args.cam)
    try:
        plan_target(args, cam)
    finally:
        if cam:
            cam.close()
    print("plan OK (arm not touched)")


def stage_pick(args: Args) -> None:
    rec: EpisodeRecorder | None = None
    cams: dict[str, LiveCamera] = {}
    if args.record and not args.sim:
        cams = {"cam0": LiveCamera(args.cam), "cam1": LiveCamera(args.cam2)}
        cam = cams["cam0"]
    else:
        cam = None if args.sim else Camera(args.cam)
    target, _, yaw, tilt = plan_target(args, cam)
    planner = Planner()
    arm = Arm(args.sim, args.channel, args.hz)
    if cams:
        rec = EpisodeRecorder(Path(args.record), cams, arm.state7, lambda: arm.last_cmd, args.task, args.fps)
        rec.start()
        print(f"recording to {args.record} at {args.fps} fps")
    H = np.array(json.loads(CALIB.read_text())["H"])
    try:
        arm.set_grip(GRIP_OPEN, 1.0)
        q_above, yaw, tilt = planner.ik(target[0], target[1], Z_TRAVEL, arm.q(), yaw, tilt)
        print("moving above the bottle")
        goto_joint(arm, planner, q_above, 4.0)
        cmd = target
        zg_h = args.z_grasp if args.z_grasp is not None else Z_GRASP
        if args.servo and isinstance(cam, LiveCamera):
            print("visual check next to the object")
            if (HERE / "cameras.json").exists():
                cmd = visual_correct_3d(arm, planner, cam, target, yaw, tilt, zg_h + 0.05)
            else:
                cmd = visual_correct(arm, planner, cam, H, target, yaw, tilt)
            move_cartesian(arm, planner, cmd[0], cmd[1], Z_TRAVEL, yaw, tilt, 2.5)  # back up above the (corrected) object
        x, y = float(cmd[0]), float(cmd[1])
        if args.grasp == "side":
            zg = planner.topdown(x, y, zg_h, yaw, tilt)[:3, 2]  # gripper pointing direction (down/forward)
            pre = np.array([x, y, zg_h]) - SIDE_APPROACH * zg
            print("moving to the pre-grasp behind the bottle")
            move_cartesian(arm, planner, pre[0], pre[1], pre[2], yaw, tilt, 3.0)
            settle(arm, planner, pre[0], pre[1], pre[2], yaw, tilt)
            print("sliding the fingers around the body")
            move_cartesian(arm, planner, x, y, zg_h, yaw, tilt, 3.0)
        else:
            print("descending")
            move_cartesian(arm, planner, x, y, zg_h + 0.03, yaw, tilt, 3.0)
            settle(arm, planner, x, y, zg_h + 0.03, yaw, tilt)
            move_cartesian(arm, planner, x, y, zg_h, yaw, tilt, 1.5)
        p = settle(arm, planner, x, y, zg_h, yaw, tilt)
        print(f"fingertips at {np.round(p, 3)}; closing gripper")
        if args.contact_close:
            g = close_until_contact(arm, squeeze=args.squeeze)
            print(f"gripper holding at {g:.2f} ({'holding something' if g > 0.12 else 'closed on nothing'})")
        else:
            arm.set_grip(GRIP_CLOSED_ON_BOTTLE, 2.0)
            time.sleep(0.5)
            g = arm.state7()[6]
            print(f"gripper settled at {g:.2f} ({'holding something' if g > GRIP_CLOSED_ON_BOTTLE + 0.05 else 'closed on nothing'})")
        print("lifting")
        move_cartesian(arm, planner, x, y, Z_TRAVEL, yaw, tilt, 3.0)
        print("holding ... (bottle picked up)")
        time.sleep(3.0)
        print("setting it back down")
        move_cartesian(arm, planner, x, y, zg_h + 0.01, yaw, tilt, 3.0)
        settle(arm, planner, x, y, zg_h + 0.005, yaw, tilt)
        arm.set_grip(GRIP_OPEN, 1.5)
        if args.grasp == "side":
            move_cartesian(arm, planner, pre[0], pre[1], pre[2], yaw, tilt, 2.0)
        move_cartesian(arm, planner, x, y, Z_TRAVEL, yaw, tilt, 2.5)
        print("returning to rest")
        goto_joint(arm, planner, REST, 5.0)
    finally:
        if rec:
            rec.stop()
        arm.close()
        for lc in cams.values():
            lc.close()
        if cam and not cams:
            cam.close()
    if rec:
        out = rec.save()
        print(f"saved LeRobot dataset: {out} ({len(rec.states)} frames)")


def main(args: Args) -> None:
    if args.detect:
        stage_detect(args)
    elif args.calibrate:
        stage_calibrate(args)
    elif args.plan:
        stage_plan(args)
    elif args.pick:
        stage_pick(args)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main(tyro.cli(Args))
