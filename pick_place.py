"""Generic pick-and-place on the YAM arm: survey the desk, plan a safe grasp, pick, hold 5 s, set down 5 cm away.

    python pick_place.py --background               # camera only, TABLE MUST BE EMPTY: the reference frame
    python pick_place.py --survey                   # camera only: every object on the table (xy, size, grasp mode)
    python pick_place.py --plan                     # camera only: the full 4-step plan -> captures/plan.jpg, plan.json
    python pick_place.py --plan --object 1 --place-dir 90
    python pick_place.py --run --object 0           # ARM MOVES: looks, plans, runs, records - one command
    python pick_place.py --run --no-rescan --record None           # ... without the mid-run stops or the dataset

The four steps of the workflow, in order:

  1. IDENTIFY   cam 0 minus the empty-table background -> one blob per object. The lowest full-width row of
                a blob is where it touches the table, so its ray hits z=0 at the front contact point; the
                silhouette width there gives the diameter and the top row gives the height (scene3d.py).
  2. TRAJECTORY every other object becomes a no-go cylinder (its radius + MARGIN, its own height + 1 cm).
                Approach azimuth, wrist yaw and travel height are searched for the largest clearance, then
                the whole path is swept in MuJoCo: every arm/gripper mesh vertex - and, once grasped, the
                carried object - must stay outside every cylinder. Nothing moves until that check passes.
  3. DECISION   flat/low or very tall -> HORIZONTAL (wrist tilted 60 deg, the fingers come in from the side
                and pinch across the body). Upright and narrow enough to straddle -> VERTICAL (straight
                down). If the chosen mode has no collision-free trajectory the other one is tried, and the
                fallback is reported.
  4. PICK/PLACE close on the object (contact-sensed, so its width does not have to be known), lift to travel
                height, hold for --hold seconds, move --place-dist (5 cm) along the clearest direction, set
                it down, release, retreat, return to rest.

--run needs no other stage first: it looks, plans, and only then moves. It stops twice more on the way -
once hovering above the object before it descends, once holding the object before it carries it across -
and re-reads the table each time with the arm's own silhouette projected out of the frame (anything hidden
behind the arm keeps its last known position). Obstacles are updated, everything still ahead is re-checked,
and a blocked place point is replaced by a free one. If nothing safe is left the arm puts the object back
where it found it and parks. Every run is recorded as a LeRobot v2.1 dataset (--record None turns it off),
including partial episodes, so a failed attempt is still on file.

Frames: robot base, x forward, y left, z up, table z=0 (m). A waypoint's "z" is the fingertip height
(grasp_site = the distal facet of the tips). Gripper: 1.0 = open (9.5 cm), 0.0 = closed.

Prerequisites: cameras.json (calib3d.py --solve) and captures/bg_cam0.png from the CURRENT session
(same lighting and camera pose, empty table). Keep the arm at rest while surveying: it is not in the
background frame, so a parked arm inside the table ROI is seen as one more object.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mujoco
import numpy as np
import tyro

import yam_mac  # noqa: F401  (macOS CAN shim; must precede i2rt imports)
from calib3d import CameraModel, load_cameras
from lerobot_record import EpisodeRecorder, LiveCamera
from dance import READY  # the upright, retracted pose the arm unfolds through on its way out
from pick_bottle import (
    GRIP_OPEN,
    REST,
    SERVO_OFFSET,
    SIDE_APPROACH,
    SIDE_TILT,
    Arm,
    Planner,
    close_until_contact,
    goto_joint,
    settle,
    visual_correct_3d,
)
from scene3d import BACKGROUND, TABLE_ROI_CAM0, foreground, grab, object_from_blob
from live_map import LiveState

HERE = Path(__file__).resolve().parent
CAPTURES = HERE / "captures"

# ---- gripper geometry (linear_4310: two 47.5 mm slides) -----------------------------------------
JAW_STROKE = 0.095  # m of opening at grip command 1.0
MAX_OBJ_WIDTH = 0.085  # m: widest object the jaw can still close on with a finger margin
JAW_GAP = 0.012  # m per side that the jaw opens wider than the object before closing
# ---- the target itself: never nudge it ------------------------------------------------------------
TALL_H = 0.08  # m: taller than this an object tips over easily
TARGET_TOL = 0.01  # m every part of the arm but the two fingertips keeps from the target's body
TALL_TOL = 0.02  # m ... for a tall object
TALL_JAW_GAP = 0.018  # m per side the jaw opens around a tall object (position error tolerance)
GOOD_AZ = -20.0  # deg: side grasps approaching at or beyond this azimuth (robot frame) came off 12 of 13 times;
# shallower approaches (-11 .. +11 deg) only 4 of 13 - most likely a gripper-orientation mismatch between the model
# and the real arm that grows towards the robot's left. They stay available as a fallback.
SLIDE_SHORT = 0.0  # m: a side grasp's fingertips stop this far before the object's estimated centre. 1.5 cm was
# tried against depth errors; the two picks that used it (with good three-camera positions) both closed beside the
# object, while the full slide completed whenever the position was right - so back to the full slide.
TARGET_H_FLOOR = 0.12  # m: the target's no-touch body is modelled at least this tall. The survey under-reads
# a shiny top on the white sheet (the same 11 cm bottle came back 6.5-11.5 cm), so its height is not
# trusted for safety: every target gets the tall tolerance, the wide jaw and the raised visual check.


def safe_height(obj: "Obj") -> float:
    if obj.measured:  # measured by hand (--given), not read off a silhouette: the floor guards a misread cap
        return obj.height
    return max(obj.height, TARGET_H_FLOOR)
# ---- workspace ----------------------------------------------------------------------------------
MARGIN = 0.025  # m of clearance to keep from every object we are not grasping
Z_TRAVEL_MIN = 0.20  # m, fingertip height while moving over an empty table
Z_TRAVEL_MAX = 0.34  # m, beyond this the arm is folded up near its joint limits
Z_CLEAR = 0.06  # m that the carried object keeps above the tallest thing it passes over
Z_TIP_MIN = 0.012  # m, never command the fingertips lower than this (the table is at 0)
REACH = (0.18, 0.55)  # m from the base: the annulus the arm works in comfortably
SHUTTLE_R = 0.33  # m: --shuttle sets the object down outward (+x) nearer the base than this, back (-x) beyond it
# ---- grasp-mode decision (step 3) ---------------------------------------------------------------
H_TOP_MIN = 0.045  # m: below this there is no band above the table to clamp from above
RATIO_MIN = 0.6  # height/diameter below this the object is "flat" (wide and low)
H_TOP_MAX = 0.16  # m: above this, reaching over the top is tippy; clamp the body from the side instead
Z_SIDE_GRASP = 0.02  # m: a side grasp closes this high - at the bottom of the object, where it is steadiest
# ---- detection ----------------------------------------------------------------------------------
MIN_OBJ_HEIGHT = 0.008  # m: a sheet of paper is not something to pick; a phone (~9 mm) is
MAX_OBJ_HEIGHT = 0.35  # m: taller than this inside the table ROI is the arm itself, or a person
MAX_OBJ_DIAM = 0.25  # m
# ---- the taped workspace zone (zone.json, from --zone) ---------------------------------------------
ZONE_FILE = HERE / "zone.json"
DET = "cam1"  # the camera that sees the whole zone (overhead, in front of the robot): it drives detection


def load_zone() -> np.ndarray | None:
    """The zone's corners in the robot frame (N x 2), or None when no zone has been set up."""
    if not ZONE_FILE.exists():
        return None
    return np.array(json.loads(ZONE_FILE.read_text())["polygon"], dtype=float)


def zone_margin(xy, zone: np.ndarray) -> float:
    """Signed distance from xy to the zone's border: positive inside, negative outside."""
    return float(cv2.pointPolygonTest(zone.astype(np.float32), (float(xy[0]), float(xy[1])), True))


# ============================================================================== step 1: identify
@dataclass
class Obj:
    name: str
    xy: tuple[float, float]  # footprint centre (the object's vertical axis) in the robot frame
    diameter: float
    height: float
    color: tuple[int, int, int]
    contour: np.ndarray | None = field(repr=False, default=None)
    measured: bool = False  # from --given: its height is trusted, not floored to TARGET_H_FLOOR

    @property
    def radius(self) -> float:
        return self.diameter / 2

    @property
    def reach(self) -> float:
        return float(np.hypot(*self.xy))

    @property
    def flatness(self) -> float:
        """height / diameter: below 1 the object is flat and wide, well above 1 it is slim and upright."""
        return self.height / max(self.diameter, 1e-6)

    def describe(self) -> str:
        return (f"{self.name:<9} xy ({self.xy[0]:+.3f}, {self.xy[1]:+.3f})  d {self.diameter * 100:5.1f} cm"
                f"  h {self.height * 100:5.1f} cm  h/d {self.flatness:4.1f}  reach {self.reach:.2f} m"
                f"  rgb {tuple(self.color)}")


def table_foreground(frame: np.ndarray, bg: np.ndarray, quiet: bool = False) -> np.ndarray:
    """`scene3d.foreground` plus symmetric lighting suppression.

    Its shadow rule drops table pixels that went *darker* than the background. But a webcam on auto
    exposure re-meters the moment anything is placed on the desk: here half the tabletop came back
    BRIGHTER and washed out (saturation 137 -> 16) and read as foreground, welding the chair, the table
    edge and the bottle into one 394x720 blob. Brightness and saturation both move under a light or
    exposure change; hue barely does. So a pixel whose hue still matches the reference is the same
    surface under different light, whichever way it went - measured on this frame, that drops 96% of the
    false area and keeps 84% of the bottle.

    The limit: a grey or white object whose (noisy) hue happens to land near the table's is suppressed
    with it. A background captured under the lighting the run will actually use is the real fix, which
    is why a large suppressed fraction is reported."""
    m = foreground(frame, bg, None)
    raw = int((m > 0).sum())
    hi, hb = (cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.int16) for f in (frame, bg))
    dh = np.abs(hi[..., 0] - hb[..., 0])
    dh = np.minimum(dh, 180 - dh)  # hue is circular
    m[(dh <= 12) & (hb[..., 1] >= 30)] = 0  # unsaturated reference pixels have no meaningful hue
    # ... which is exactly the white zone: there a shadow is the same grey-white only darker, and an
    # exposure re-meter makes it brighter. Keep a pixel only if it gained colour or went nearly black; a
    # grey or white object on the sheet is lost with it.
    paper = (hb[..., 1] < 45) & (hb[..., 2] > 140)
    shade = paper & (hi[..., 1] < 50) & (hi[..., 2] > 0.35 * hb[..., 2])
    m[shade] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    kept = int((m > 0).sum())
    if raw > 20000 and kept < 0.6 * raw and not quiet:
        print(f"   (note: {100 * (1 - kept / raw):.0f}% of the changed pixels were the table itself under "
              "different light - re-run --background if detection looks wrong)")
    return m


def trim_tail(sil: np.ndarray) -> np.ndarray:
    """Cut a narrow tail off the bottom of a silhouette.

    The camera looks at the table almost along the light, so an object's shadow runs towards the lens
    and joins its silhouette. `object_from_blob` takes the lowest row that is at least a quarter of the
    body's width as the table contact, and a shadow wider than that drags the contact - and so the whole
    footprint - a few centimetres towards the camera. The body of an object ends where its width
    collapses; a tapered cup narrows gradually and is left alone."""
    rows = np.where(sil.any(axis=1))[0]
    if len(rows) < 10:
        return sil
    widths = np.array([np.ptp(np.where(sil[v] > 0)[0]) + 1 for v in rows])
    narrow = widths < 0.5 * np.percentile(widths, 90)
    end = len(rows) - 1
    for i in range(int(np.argmax(widths)), len(rows)):  # downwards from the widest row
        if narrow[i]:
            end = max(i - 1, 0)
            break
    out = sil.copy()
    out[rows[end] + 1 :] = 0
    return out


_WS_MASK: dict[tuple, np.ndarray] = {}


def workspace_mask(cam: CameraModel, shape: tuple[int, int]) -> np.ndarray:
    """The pixels a pickable object could occupy: everything standing on the reachable ring of table,
    from the tabletop up to MAX_OBJ_HEIGHT, projected into the image.

    A rectangular pixel window cannot express this - the one scene3d uses also contains the floor, the
    chair and the wall past the far edge of the desk, and MORPH_CLOSE then welds a moved chair to the
    bottle in front of it into a single 394x600 blob. This window is the workspace itself."""
    key = (id(cam), shape)
    if key not in _WS_MASK:
        m = np.zeros(shape, np.uint8)
        zone = load_zone()
        g = np.arange(-REACH[1], REACH[1] + 1e-9, 0.01)
        if zone is not None:  # the taped zone, a hair beyond its border so an object on it is still whole
            xy = np.array([(x, y) for x in g for y in g if zone_margin((x, y), zone) >= -0.015])
        else:
            xy = np.array([(x, y) for x in g for y in g if REACH[0] - 0.02 <= np.hypot(x, y) <= REACH[1] + 0.02])
        lo = np.clip(cam.project(np.c_[xy, np.zeros(len(xy))]), -1e5, 1e5).astype(np.int32)
        hi = np.clip(cam.project(np.c_[xy, np.full(len(xy), MAX_OBJ_HEIGHT)]), -1e5, 1e5).astype(np.int32)
        for a, b in zip(lo, hi):
            cv2.line(m, tuple(a), tuple(b), 255, 5)
        # the columns are 1 cm apart in the world, which is several pixels in the near field: close the
        # gaps so the window is solid and "the blob reaches the edge of it" means what it says
        _WS_MASK[key] = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    return _WS_MASK[key]


def full_height(cam: CameraModel, sil: np.ndarray, raw: np.ndarray, h: float) -> float:
    """Re-measure an object's height with the raw changed pixels that continue it straight upwards inside
    its own column band (a cap the paper rule removed). Connectivity is judged inside the band only, so
    something behind the object (a person, the robot) joins only if it actually touches its top."""
    ys, xs = np.where(sil > 0)
    band = np.zeros_like(raw)
    band[: ys.max() + 1, xs.min() : xs.max() + 1] = raw[: ys.max() + 1, xs.min() : xs.max() + 1]
    band[sil > 0] = 255
    _, lab = cv2.connectedComponents(band)
    ext = lab == lab[ys[0], xs[0]]
    top = int(np.where(ext)[0].min())
    if top >= ys.min() or ys.min() - top > (ys.max() - ys.min()):  # nothing above, or more than doubling it
        return h
    o2 = object_from_blob(cam, ext.astype(np.uint8) * 255)
    return max(h, o2["height"]) if o2 else h


def survey(frame: np.ndarray, cam: CameraModel, min_area: float = 2000.0,
           ignore: np.ndarray | None = None, quiet: bool = False, key: str = DET) -> list[Obj]:
    """Every object on the table, with its footprint centre, diameter and height (step 1).
    `ignore` is a mask of pixels to drop - the arm's own silhouette during a mid-run rescan."""
    bg = cv2.imread(str(BACKGROUND[key]))
    if bg is None:
        raise RuntimeError(f"{BACKGROUND[key]} missing: capture an empty-table frame first")
    win = workspace_mask(cam, frame.shape[:2])
    fg = table_foreground(frame, bg, quiet)
    # the same frame without the paper/lighting suppression: a silver cap on the white sheet is as grey and
    # bright as the paper, so it is suppressed with it and the object reads short (an 11 cm bottle came
    # back 8 cm tall, below TALL_H). Heights are re-measured from this; it can only make them taller.
    raw = cv2.morphologyEx(foreground(frame, bg, None), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    raw[win == 0] = 0
    if ignore is not None:
        raw[ignore > 0] = 0
    fg[win == 0] = 0
    if ignore is not None:
        fg[ignore > 0] = 0
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objs: list[Obj] = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < min_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        sil = np.zeros_like(fg)
        cv2.drawContours(sil, [c], -1, 255, -1)
        sil = trim_tail(sil)
        ys, xs = np.where(sil > 0)
        if len(ys) < 50:
            continue
        bx, by, bw, bh = int(xs.min()), int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1
        # a blob that reaches the edge of the window is cut there: its lowest row is then not where it
        # touches the table, or its top row is not its top, so the footprint or the height is fiction.
        # A cable trailing up into the robot's base plate reads as a 20 cm tall object this way.
        edge = (bx <= 1 or by <= 1 or bx + bw >= fg.shape[1] - 1 or by + bh >= fg.shape[0] - 1
                or bool(np.any((cv2.dilate(sil, np.ones((5, 5), np.uint8)) > 0) & (win == 0))))
        if edge:
            if quiet:
                continue
            o = object_from_blob(cam, sil)
            where = f" (roughly {np.round(o['axis'], 2)})" if o else ""
            print(f"   (skipped a {cv2.contourArea(c):.0f} px blob at pixel ({bx}, {by}){where}: it runs off "
                  "the edge of the surveyed workspace, so its size cannot be measured)")
            continue
        o = object_from_blob(cam, sil)
        if o is None:
            continue
        o["height"] = full_height(cam, sil, raw, o["height"])
        if not (0.12 < float(np.hypot(*o["axis"])) < 0.70):
            continue
        zone = load_zone()
        if zone is not None and zone_margin(o["axis"], zone) < 0:
            continue  # standing outside the zone: not ours to pick
        if not (0.015 < o["diameter"] < MAX_OBJ_DIAM) or not (MIN_OBJ_HEIGHT < o["height"] < MAX_OBJ_HEIGHT):
            continue
        objs.append(Obj(
            name=f"object{len(objs)}",
            xy=(float(o["axis"][0]), float(o["axis"][1])),
            diameter=float(o["diameter"]),
            height=float(o["height"]),
            color=tuple(int(v) for v in cv2.mean(frame, sil)[:3][::-1]),
            contour=c,
        ))
    return objs


# ============================================= multi-view: carve the zone with every calibrated camera
VOXEL = 0.005  # m
CARVE_MIN_Z = 0.015  # m: occupancy below this is the sheet itself (shadows sit on it in every view)


CARVE_RAISE = 0.04  # m: the most carving may raise an object above the detection camera's own height
CARVE_LAST: dict = {}  # the last carve's solid columns: {"xy": (N, 2), "height": (N,)}


def carve(frames: dict[str, np.ndarray], cams: dict[str, CameraModel],
          ignore: dict[str, np.ndarray] | None = None) -> list[dict] | None:
    """Space carving over the zone: a 5 mm voxel is solid only if at least two cameras see it and every
    camera that sees it (not hidden behind the arm) shows a change there. A shadow or an exposure shift in
    one view is carved away by the others, and a cap one view loses against the paper is kept by the rest.
    Returns one blob per object: its footprint centre, footprint area, height and how many views saw it."""
    zone = load_zone()
    keys = [k for k in frames if k in cams and cv2.imread(str(BACKGROUND[k])) is not None]
    if zone is None or len(keys) < 2:
        return None
    lo, hi = zone.min(axis=0) - 0.02, zone.max(axis=0) + 0.02
    xs, ys = np.arange(lo[0], hi[0], VOXEL), np.arange(lo[1], hi[1], VOXEL)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    inside = np.array([zone_margin((x, y), zone) >= -0.015 for x, y in zip(gx.ravel(), gy.ravel())])
    cols = np.c_[gx.ravel(), gy.ravel()][inside]
    zs = np.arange(0.0025, MAX_OBJ_HEIGHT, VOXEL)
    P = np.c_[np.repeat(cols, len(zs), axis=0), np.tile(zs, len(cols))]
    yes = np.zeros(len(P), np.int16)
    no = np.zeros(len(P), bool)
    for k in keys:
        cam, frame = cams[k], frames[k]
        # only what this camera's own survey accepted as objects: raw changed pixels include every exposure
        # shift, and with two views those intersect into ghost volumes (a 30 cm "bottle", three phantoms)
        fg = np.zeros(frame.shape[:2], np.uint8)
        if k == DET:
            for o in survey(frame, cam, 1500.0, ignore=None if ignore is None else ignore.get(k), quiet=True, key=k):
                cv2.drawContours(fg, [o.contour], -1, 255, -1)
        else:
            # a helper view only has to confirm: any sizeable change in its window counts, even one cut by
            # its frame edge (cam 0 sees just the far half of the zone, so bottles' bases fall off its frame)
            raw = table_foreground(frame, cv2.imread(str(BACKGROUND[k])), quiet=True)
            raw[workspace_mask(cam, raw.shape) == 0] = 0
            if ignore is not None and ignore.get(k) is not None:
                raw[ignore[k] > 0] = 0
            n, lab, st, _ = cv2.connectedComponentsWithStats(raw)
            for c in range(1, n):
                if st[c, cv2.CC_STAT_AREA] >= 1500:
                    fg[lab == c] = 255
        fg = cv2.dilate(fg, np.ones((7, 7), np.uint8))  # a few px of calibration error must not carve a real edge
        h, w = fg.shape
        front = (P @ cam.R.T + cam.t)[:, 2] > 0.05
        uv = np.full((len(P), 2), -1.0)
        uv[front] = cam.project(P[front])
        u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        seen = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if ignore is not None and k in ignore and ignore[k] is not None:
            seen[seen] &= ignore[k][v[seen], u[seen]] == 0  # behind the arm: this view abstains
        hit = np.zeros(len(P), bool)
        hit[seen] = fg[v[seen], u[seen]] > 0
        yes += hit
        no |= seen & ~hit
    occ = ((~no) & (yes >= 2)).reshape(len(cols), len(zs))
    occ[:, zs < CARVE_MIN_Z] = False
    top = len(zs) - 1 - np.argmax(occ[:, ::-1], axis=1)  # highest solid voxel of each column
    height = np.where(occ.any(axis=1), zs[top] + VOXEL / 2, 0.0)
    views = yes.reshape(len(cols), len(zs)).max(axis=1)
    grid = np.zeros(gx.shape, np.uint8)
    idx = np.flatnonzero(inside)
    solid = occ.sum(axis=1) >= 2  # at least 1 cm of solid column
    CARVE_LAST.update(xy=cols[solid], height=height[solid] if len(height) else height)
    grid.ravel()[idx[solid]] = 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    hmap = np.zeros(gx.shape)
    hmap.ravel()[idx] = height
    vmap = np.zeros(gx.shape)
    vmap.ravel()[idx] = views
    blobs = []
    for c in range(1, n):
        cells = lab == c
        if cells.sum() < 4:  # < 1 cm^2
            continue
        blobs.append({"xy": (float(gx[cells].mean()), float(gy[cells].mean())),
                      "area": float(cells.sum() * VOXEL * VOXEL), "height": float(hmap[cells].max()),
                      "views": int(vmap[cells].max())})
    return blobs


def survey_multi(frames: dict[str, np.ndarray], cams: dict[str, CameraModel], min_area: float = 2000.0,
                 ignore: dict[str, np.ndarray] | None = None) -> list[Obj]:
    """`survey` on the detection camera for each object's outline, with its position and height taken from
    `carve` across every calibrated camera. Carved objects the detection camera missed are added."""
    objs = survey(frames[DET], cams[DET], min_area, ignore=None if ignore is None else ignore.get(DET))
    blobs = carve(frames, cams, ignore)
    if blobs is None:
        return objs
    used = set()
    for o in objs:
        ds = [float(np.hypot(*(np.array(b["xy"]) - np.array(o.xy)))) for b in blobs]
        j = int(np.argmin(ds)) if ds else -1
        if j >= 0 and ds[j] <= 0.06 and j not in used:
            used.add(j)
            b = dict(blobs[j])
            # only the solid columns near this object: a ghost merged into its blob must not drag the centre
            near = np.hypot(*(CARVE_LAST["xy"] - np.array(o.xy)).T) <= max(0.05, o.radius + 0.02)
            if near.sum() >= 4:
                b["xy"] = tuple(float(v) for v in CARVE_LAST["xy"][near].mean(axis=0))
                b["height"] = float(CARVE_LAST["height"][near].max())
            d = float(np.hypot(*(np.array(b["xy"]) - np.array(o.xy))))
            print(f"   {o.name}: {b['views']} views put it at {np.round(b['xy'], 3)} ({d * 100:.1f} cm from the "
                  f"single-camera estimate), {b['height'] * 100:.1f} cm tall")
            o.xy = b["xy"]
            # the carved top of the columns near this object only: the whole blob's top included two-view
            # ghosts (15-16 cm for a 11 cm bottle), and the detection camera alone reads short whenever the
            # parked arm's mask covers the cap (5.3 cm) - locally carved it came back 10.0 cm
            # ...but never more than CARVE_RAISE above the detection camera's own reading: something carved
            # above an object (a ghost, a hand in one view) once made a 9 cm object 27 cm tall
            o.height = max(o.height, min(b["height"], o.height + CARVE_RAISE))
        else:
            print(f"   {o.name}: not confirmed by a second camera - kept at the single-camera estimate")
    ghosts = [b for j, b in enumerate(blobs) if j not in used]
    if ghosts:
        # with two views from the same side, cones of different image regions cross in empty space: an 18 cm
        # "object" 8 cm behind the bottle. Only the detection camera's own objects are real; carving refines them.
        print(f"   (ignored {len(ghosts)} carve-only blob(s) at "
              f"{[np.round(b['xy'], 3).tolist() for b in ghosts]}: two-view ghosts, not confirmed as objects)")
    return objs


CAM_MOVED_PX = 3.0  # px: a camera whose fixed background has shifted more than this no longer fits its calibration


def camera_shift(key: str, frame: np.ndarray) -> float:
    """How far (px) this camera's view has shifted since its empty-zone background, measured on fixed
    background features (ORB matches, median displacement). A bumped camera still sees the table but its
    calibration no longer does: cam 0 moved 81 px and cam 2 11 px unnoticed and three picks missed."""
    bg = cv2.imread(str(BACKGROUND[key]), cv2.IMREAD_GRAYSCALE)
    if bg is None:
        return float("inf")
    orb = cv2.ORB_create(3000)
    ka, da = orb.detectAndCompute(bg, None)
    kb, db = orb.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None:
        return float("inf")
    m = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db), key=lambda x: x.distance)[:400]
    if len(m) < 30:
        return float("inf")
    d = np.array([np.subtract(kb[x.trainIdx].pt, ka[x.queryIdx].pt) for x in m])
    return float(np.linalg.norm(np.median(d, axis=0)))


def still_cameras(frames: dict[str, np.ndarray], models: dict) -> dict:
    """The calibrated cameras that have not moved. A moved helper is dropped from the fusion for this run;
    a moved detection camera stops the run - nothing it sees can be trusted."""
    keep = {}
    for k, f in frames.items():
        sh = camera_shift(k, f)
        if sh <= CAM_MOVED_PX:
            keep[k] = models[k]
        else:
            print(f"   !! {k} has moved {sh:.0f} px since its calibration/background - left out of the fused workspace")
    if DET not in keep:
        raise RuntimeError(f"the detection camera {DET} has moved since its calibration: recalibrate before running")
    return keep


def grab_all(args: "Args") -> dict[str, np.ndarray]:
    """One frame from every calibrated camera (camN is OpenCV index N)."""
    return {k: grab(int(k[3:])) for k in load_cameras()}


# ========================================================== planner with a free approach azimuth
class AzPlanner(Planner):
    """Planner whose approach azimuth can be decoupled from the target's bearing from the base.

    `Planner.topdown` always tilts the wrist in the plane through the base and the target, so a side
    grasp can only ever come in radially. Overriding it as an instance method makes every inherited
    helper (ik, move_cartesian, settle, visual_correct_3d) honour `self.az` instead."""

    def __init__(self) -> None:
        super().__init__()
        self.az: float | None = None  # None = radial, i.e. exactly Planner.topdown
        self._jaw_cached: np.ndarray | None = None

    def topdown(self, x: float, y: float, z: float, yaw: float, tilt: float = 0.0) -> np.ndarray:
        if self.az is None:
            return Planner.topdown(x, y, z, yaw, tilt)
        # the orientation only depends on the azimuth, so build the frame for a target sitting on the
        # desired azimuth and then move it to (x, y, z)
        T = Planner.topdown(float(np.cos(self.az)), float(np.sin(self.az)), z, yaw, tilt)
        T[:3, 3] = (x, y, z)
        return T

    def frame_at(self, xy: tuple[float, float], yaw: float, tilt: float, az: float | None) -> np.ndarray:
        keep, self.az = self.az, az
        try:
            return self.topdown(xy[0], xy[1], 0.0, yaw, tilt)[:3, :3]
        finally:
            self.az = keep

    def approach_dir(self, xy: tuple[float, float], yaw: float, tilt: float, az: float | None) -> np.ndarray:
        """Unit vector the gripper points along: the fingers travel this way towards the object."""
        return self.frame_at(xy, yaw, tilt, az)[:, 2]

    def jaw_dir(self, xy: tuple[float, float], yaw: float, tilt: float, az: float | None) -> np.ndarray:
        """Unit vector along which the two fingertips separate, in the robot frame."""
        return self.frame_at(xy, yaw, tilt, az) @ self.jaw_local

    @property
    def jaw_local(self) -> np.ndarray:
        """Fingertip separation axis in the grasp_site frame, measured once from the model."""
        if self._jaw_cached is None:
            m, d = self.model, self.data
            d.qpos[:] = 0
            for j in ("joint7", "joint8"):  # slide both tips open
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                if jid >= 0:
                    d.qpos[m.jnt_qposadr[jid]] = 0.03
            mujoco.mj_kinematics(m, d)
            bl, br = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("tip_left", "tip_right"))
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
            v = d.xpos[br] - d.xpos[bl]
            self._jaw_cached = d.site_xmat[sid].reshape(3, 3).T @ (v / np.linalg.norm(v))
        return self._jaw_cached


class Workspace:
    """Swept-volume clearance of the arm (and whatever it carries) against the objects on the table."""

    def __init__(self, planner: AzPlanner, obstacles: list[Obj], margin: float = MARGIN):
        self.planner, self.obstacles, self.margin = planner, obstacles, margin
        self.target: tuple | None = None  # (xy, radius, height, tolerance) of the object being picked
        m = planner.model
        self.geoms: list[tuple[int, np.ndarray]] = []
        self.tips: list[np.ndarray] = []
        for gi in range(m.ngeom):
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gi])
            if body in ("world", "base"):  # the base never moves: nothing we could steer around it
                continue
            mid = m.geom_dataid[gi]
            if mid < 0:
                continue
            v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            verts = np.asarray(m.mesh_vert[v0 : v0 + nv], dtype=np.float64)
            if len(verts) > 400:  # a few hundred surface points are plenty at a 2.5 cm margin
                verts = verts[:: len(verts) // 400]
            self.geoms.append((gi, verts))
            self.tips.append(np.full(len(verts), body.startswith("tip")))
        # the jaw is open for most of the approach, so sweep it open: the tips then stand 4.75 cm out
        # on either side, which is the widest the gripper ever is
        self.is_tip = np.concatenate(self.tips)  # aligned with points(): True for fingertip vertices
        self.jaw_qpos = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                         for j in ("joint7", "joint8")
                         if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) >= 0]

    def geom_points(self, q6: np.ndarray) -> list[np.ndarray]:
        d = self.planner.data
        d.qpos[:] = 0
        d.qpos[:6] = q6
        for a in self.jaw_qpos:
            d.qpos[a] = 0.0475
        mujoco.mj_kinematics(self.planner.model, d)
        return [v @ d.geom_xmat[gi].reshape(3, 3).T + d.geom_xpos[gi] for gi, v in self.geoms]

    def points(self, q6: np.ndarray) -> np.ndarray:
        d = self.planner.data
        d.qpos[:] = 0
        d.qpos[:6] = q6
        for a in self.jaw_qpos:
            d.qpos[a] = 0.0475
        mujoco.mj_kinematics(self.planner.model, d)
        return np.concatenate([v @ d.geom_xmat[gi].reshape(3, 3).T + d.geom_xpos[gi] for gi, v in self.geoms])

    def clearance(self, pts: np.ndarray) -> tuple[float, str]:
        """Smallest gap between any point and any object's no-go cylinder (negative = inside it)."""
        worst, who = 1e9, ""
        for o in self.obstacles:
            low = pts[:, 2] < o.height + 0.01  # points above an object sail over it
            if not low.any():
                continue
            gap = float(np.linalg.norm(pts[low, :2] - np.array(o.xy), axis=1).min()) - o.radius - self.margin
            if gap < worst:
                worst, who = gap, o.name
        return worst, who

    def target_gap(self, pts: np.ndarray) -> float:
        """Clearance of the arm - everything but the fingertips, which have to reach around it - from the
        target's own body, inflated by its tolerance. The target is not in `obstacles`, so without this the
        gripper housing could brush a tall object on the way in or out and tip it over."""
        if self.target is None:
            return 1e9
        xy, r, h, tol = self.target
        sel = ~self.is_tip & (pts[: len(self.is_tip), 2] < h + 0.01)
        if not sel.any():
            return 1e9
        return float(np.linalg.norm(pts[: len(self.is_tip)][sel, :2] - np.asarray(xy), axis=1).min()) - r - tol

    def scan(self, q_from: np.ndarray, q_to: np.ndarray, carry=None, n: int = 12) -> tuple[float, str, float]:
        """Sweep the straight joint-space path the arm will actually glide through: worst clearance from
        the objects, what it was against, and the lowest point any mesh reaches (the table check).

        `Planner.path_is_safe` tests geom origins against a fixed 2 cm floor, which vetoes a legitimate
        low grasp on a flat object; the real fingertip surface is a mesh vertex, so it is measured here."""
        worst, who, zmin = 1e9, "", 1e9
        for a in np.linspace(0.0, 1.0, n):
            q = (1 - a) * q_from + a * q_to
            pts = self.points(q)
            zmin = min(zmin, float(pts[:, 2].min()))  # arm only: the carried object legitimately
            if carry is not None:                     # starts and ends sitting on the table
                pts = np.vstack([pts, carry(self.planner.fk_pos(q))])
            gap, w = self.clearance(pts)
            if gap < worst:
                worst, who = gap, w
            tg = self.target_gap(pts)
            if tg < worst:
                worst, who = tg, "the target itself"
        return worst, who, zmin

    def aim(self, obj: Obj | None, xy=None) -> None:
        """Where the target stands now (None while it hangs in the gripper)."""
        if obj is None:
            self.target = None
        else:
            h = safe_height(obj)
            tol = TALL_TOL if h > TALL_H else TARGET_TOL
            self.target = (tuple(xy if xy is not None else obj.xy), obj.radius, h, tol)


def arm_mask(ws: Workspace, cam: CameraModel, q6: np.ndarray, shape: tuple[int, int],
             dilate: int = 27) -> np.ndarray:
    """The pixels the arm itself covers, so a mid-run scan does not read the robot as an object.
    Anything hiding behind this mask keeps whatever the previous scan knew about it."""
    m = np.zeros(shape, np.uint8)
    for pts in ws.geom_points(q6):
        front = (pts @ cam.R.T + cam.t)[:, 2] > 0.05  # points behind the lens project nonsensically
        if front.sum() < 3:
            continue
        px = cam.project(pts[front])
        px = px[np.isfinite(px).all(axis=1) & (np.abs(px) < 1e4).all(axis=1)]
        if len(px) < 3:
            continue
        cv2.fillConvexPoly(m, cv2.convexHull(px.astype(np.int32)), 255)
    return cv2.dilate(m, np.ones((dilate, dilate), np.uint8))


def carried_points(obj: Obj, z_grasp: float, ahead=(0.0, 0.0)):
    """Surface points of the object while it hangs in the gripper, given the fingertip position.
    It was grasped `z_grasp` above its base, so it sticks out that far below the fingertips, and its axis
    is `ahead` (xy) beyond the fingertips (a side grasp stops SLIDE_SHORT before the centre)."""
    th = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ring = np.stack([np.cos(th), np.sin(th)], axis=1) * (obj.radius + 0.005) + np.asarray(ahead, dtype=float)

    def f(tip: np.ndarray) -> np.ndarray:
        lo = max(tip[2] - z_grasp, 0.0)
        xy = ring + tip[:2]
        return np.array([[p[0], p[1], z] for z in np.linspace(lo, lo + obj.height, 4) for p in xy])

    return f


def grasp_ahead(planner: "AzPlanner", plan: "Plan") -> np.ndarray:
    """Where the object's axis sits relative to the fingertips once grasped (xy): SLIDE_SHORT along the
    approach for a side grasp, nothing for a top grasp."""
    if plan.mode != "side":
        return np.zeros(2)
    keep = planner.az
    zg = planner.approach_dir(plan.pick, plan.yaw, plan.tilt, plan.az)[:2]
    planner.az = keep
    return SLIDE_SHORT * zg / (np.linalg.norm(zg) + 1e-9)


def table_floor(z_from: float, z_to: float) -> float:
    """Lowest the arm's meshes may reach on a leg: 1 cm normally, and just under the commanded
    fingertip height when that leg deliberately goes low (a flat object is grasped at ~1.2 cm)."""
    return min(0.01, min(z_from, z_to) - 0.004)


def follow(arm: Arm, planner: AzPlanner, ws: "Workspace", qs: list, seconds: float, carry=None,
           floor_z: float | None = None) -> None:
    """Glide through a joint path `validate` produced, re-checking each segment against the obstacles, the
    target and the table as it goes (the scene may have been updated since). The first segment is scanned
    from the MEASURED joints, because that is where `Arm.glide` starts; after `settle` the commanded pose
    is deliberately off (it compensates sag) and the model puts it lower than the arm really is."""
    q = arm.q()
    for q_next in qs:
        gap, who, zmin = ws.scan(q, np.asarray(q_next, dtype=float), carry)
        if floor_z is not None and zmin < floor_z:
            raise RuntimeError(f"a planned segment would reach {zmin * 100:.1f} cm: too low")
        if gap < 0:
            raise RuntimeError(f"a planned segment would enter {who}")
        arm.glide(np.asarray(q_next, dtype=float), seconds / len(qs))
        q = np.asarray(q_next, dtype=float)


def move_line(arm: Arm, planner: AzPlanner, ws: "Workspace", xyz, yaw: float, tilt: float, seconds: float,
              carry=None, step: float = 0.02, qs: list | None = None) -> None:
    """`pick_bottle.move_cartesian` with the swept-mesh table check instead of the fixed 2 cm floor.

    IK is seeded from the last COMMANDED joints, as `validate` chains it: seeding from the measured joints
    (which sag a little under load) put IK on another branch mid-leg and failed a step the plan had passed."""
    if qs:  # the leg was validated: follow exactly those joints, never a fresh IK solution
        here = planner.fk_pos(arm.q())
        follow(arm, planner, ws, qs, seconds, carry, table_floor(here[2], float(xyz[2])) - 0.004)
        return
    # the line starts where the fingertips ARE: after `settle` the commanded pose is deliberately off the
    # waypoint (it compensates sag), and sweeping from it made a straight lift look like it hit the table.
    # So: the measured tip position, solved on the commanded pose's IK branch.
    p0, p1 = planner.fk_pos(arm.q()), np.array(xyz, dtype=float)
    try:
        q, _, _ = planner.ik(p0[0], p0[1], p0[2], np.array(arm.last_cmd[:6], dtype=float), yaw, tilt)
    except RuntimeError:
        q = arm.q()
    n = max(1, int(np.linalg.norm(p1 - p0) / step))
    floor = table_floor(p0[2], p1[2])
    for i in range(1, n + 1):
        p = p0 + (p1 - p0) * i / n
        q_next, _, _ = planner.ik(p[0], p[1], p[2], q, yaw, tilt)
        gap, who, zmin = ws.scan(q, q_next, carry)
        if zmin < floor:
            raise RuntimeError(f"segment towards {np.round(p, 3)} would reach {zmin * 100:.1f} cm: too low")
        if gap < 0:
            raise RuntimeError(f"segment towards {np.round(p, 3)} would enter {who}")
        arm.glide(q_next, seconds / n)
        q = q_next


def fmt_gap(g: float) -> str:
    """Clearances come back as a 1e9 sentinel when there is nothing to be clear of."""
    return "no other objects" if g > 1e6 else f"{g * 100:+.1f} cm"


def free_gap(xy, obstacles: list[Obj], extra: float = 0.0) -> tuple[float, str]:
    """Analytic clearance of a point (inflated by `extra`) from the no-go cylinders; a fast pre-filter."""
    worst, who = 1e9, ""
    for o in obstacles:
        gap = float(np.linalg.norm(np.asarray(xy) - np.array(o.xy))) - o.radius - extra - MARGIN
        if gap < worst:
            worst, who = gap, o.name
    return worst, who


# ============================================================================== step 3: decision
def decide_grasp(obj: Obj) -> tuple[str, str]:
    """VERTICAL (top-down) or HORIZONTAL (side), from the object's flatness and height."""
    h, d, f = obj.height, obj.diameter, obj.flatness
    if d > MAX_OBJ_WIDTH:
        raise RuntimeError(f"{obj.name} is {d * 100:.1f} cm wide, the jaw only spans "
                           f"{MAX_OBJ_WIDTH * 100:.0f} cm (pick another --object)")
    if h < H_TOP_MIN:
        return "side", (f"low object ({h * 100:.1f} cm < {H_TOP_MIN * 100:.0f} cm): closing from above would "
                        "drive the fingertips into the table")
    if f < RATIO_MIN:
        return "side", (f"flat object (h/d {f:.2f} < {RATIO_MIN}): no upright band to clamp, so pinch across "
                        "it from the side")
    if h > H_TOP_MAX:
        return "side", (f"tall object ({h * 100:.1f} cm > {H_TOP_MAX * 100:.0f} cm): clamp the body at "
                        "mid-height instead of reaching over the top")
    return "top", f"upright and narrow enough to straddle (h {h * 100:.1f} cm, h/d {f:.2f}): descend along its axis"


def grasp_height(mode: str, obj: Obj) -> float:
    """Fingertip height at the moment of closing."""
    if mode == "top":
        return float(np.clip(0.45 * obj.height, 0.02, 0.09))
    return float(np.clip(min(Z_SIDE_GRASP, 0.5 * obj.height), Z_TIP_MIN, 0.10))


def travel_height(obj: Obj, obstacles: list[Obj], z_grasp: float) -> float:
    """High enough that the carried object (which hangs z_grasp below the fingertips) clears everything,
    and that the fingertips hovering over the target on the way in clear its own top (they are exempt from
    the target clearance check, so a 25 cm wash bottle under a 20 cm hover was never flagged)."""
    tallest = max([o.height for o in obstacles] + [safe_height(obj) - z_grasp, 0.0])
    return float(min(max(Z_TRAVEL_MIN, tallest + Z_CLEAR + z_grasp), Z_TRAVEL_MAX))


# ============================================================================ step 2: trajectory
@dataclass
class WP:
    """One commanded waypoint. xyz is the fingertip target; `grip` is commanded after arriving."""
    label: str
    xyz: tuple[float, float, float]
    seconds: float = 2.5
    kind: str = "line"  # line (cartesian) | pose (explicit q) | joint (ik on xyz) | close | hold | rescan | rest
    grip: float | None = None
    settle: bool = False
    q: np.ndarray | None = None  # for kind "pose": the joint target, xyz being only its FK position
    qs: list | None = None  # the joint path `validate` checked for this leg; `execute` follows exactly this


@dataclass
class Plan:
    obj: Obj
    obstacles: list[Obj]
    mode: str
    reason: str
    yaw: float
    tilt: float
    az: float | None
    pick: tuple[float, float]
    place: tuple[float, float]
    z_grasp: float
    z_travel: float
    open_grip: float
    waypoints: list[WP] = field(default_factory=list)
    clearance: float = 0.0
    tight_at: str = ""
    legs: list[tuple[str, float]] = field(default_factory=list)
    low_point: float = 0.0
    margin: float = MARGIN
    servo_ok: bool = True
    servo_note: str = ""
    servo_dist: float = SERVO_OFFSET  # m from the pick point to the visual-check point, towards the camera
    lead: list = field(default_factory=list)  # validated legs out of the visual check to above the object
    servo_dir: tuple | None = None  # unit xy direction from the pick point to the check point

    def to_json(self) -> dict:
        return {
            "object": {"name": self.obj.name, "xy": list(self.obj.xy), "diameter": self.obj.diameter,
                       "height": self.obj.height, "flatness": self.obj.flatness},
            "obstacles": [{"name": o.name, "xy": list(o.xy), "diameter": o.diameter, "height": o.height}
                          for o in self.obstacles],
            "grasp": {"mode": self.mode, "reason": self.reason, "yaw": self.yaw, "tilt": self.tilt,
                      "approach_azimuth": self.az, "z_grasp": self.z_grasp, "open_grip": self.open_grip},
            "pick": list(self.pick), "place": list(self.place), "z_travel": self.z_travel,
            "clearance_m": self.clearance, "tightest_against": self.tight_at, "low_point_m": self.low_point,
            "visual_check": {"enabled": self.servo_ok, "note": self.servo_note},
            "waypoints": [{"label": w.label, "xyz": list(w.xyz), "kind": w.kind, "grip": w.grip,
                           "seconds": w.seconds, "clearance_m": dict(self.legs).get(w.label)}
                          for w in self.waypoints],
        }


def build_waypoints(planner: AzPlanner, obj: Obj, mode: str, yaw: float, tilt: float, az: float | None,
                    pick: tuple[float, float], place: tuple[float, float], z_grasp: float, z_travel: float,
                    open_grip: float, hold: float, pause: float = 0.0, park: str = "rest") -> list[WP]:
    x, y = pick
    px, py = place
    # the arm stops and re-reads the table at the two points where it is about to commit to something:
    # just before it descends on the object, and just before it carries it across the desk
    look1 = [WP("look at the scene again", (x, y, z_travel), pause, kind="rescan")] if pause > 0 else []
    look2 = list(look1)
    # unfolding straight from the folded rest pose sweeps the forearm low across the table; going
    # through READY (upright and retracted, the pose dance.py rises to) keeps that sweep near the base
    ready = tuple(float(v) for v in planner.fk_pos(READY))
    wps = [WP("unfold to ready", ready, 3.0, kind="pose", q=np.array(READY), grip=open_grip),
           WP("above the object", (x, y, z_travel), 4.0, kind="joint")] + look1
    if mode == "side":
        # the fingers point outward-and-down; back off along that axis for the pre-grasp
        zg = planner.approach_dir(pick, yaw, tilt, az)
        # the fingertips stop SLIDE_SHORT before the estimated centre: the object's position is least certain
        # along the front cameras' line of sight, and a slide to the full depth pushed a bottle 1.8 and 2.8 cm
        # with the palm. The jaw's extra opening (TALL_JAW_GAP per side) still takes it; everything after the
        # grasp (lift, carry, lower, release) is shifted by the same amount, so the object lands where planned.
        short = SLIDE_SHORT * np.r_[zg[:2] / (np.linalg.norm(zg[:2]) + 1e-9), 0.0]
        x, y = x - short[0], y - short[1]
        px, py = px - short[0], py - short[1]
        pre = tuple(np.array([x, y, z_grasp]) - SIDE_APPROACH * zg)
        pre_place = tuple(np.array([px, py, z_grasp]) - SIDE_APPROACH * zg)
        wps += [
            WP("pre-grasp beside the object", pre, 3.0, settle=True),
            WP("slide the fingers around it", (x, y, z_grasp), 3.0, settle=True),
            WP("close on the object", (x, y, z_grasp), 0.0, kind="close"),
            WP("lift", (x, y, z_travel), 3.0),
            WP(f"hold {hold:.0f} s", (x, y, z_travel), hold, kind="hold"),
        ] + look2 + [
            WP("carry to the place point", (px, py, z_travel), 3.5),
            WP("lower", (px, py, z_grasp), 3.0, settle=True),
            WP("release", (px, py, z_grasp), 1.5, grip=GRIP_OPEN),
            WP("back the fingers out", pre_place, 2.5),
            WP("clear of the table", (px, py, z_travel), 2.5),
        ]
    else:
        wps += [
            WP("above the grasp height", (x, y, z_grasp + 0.03), 3.0, settle=True),
            WP("descend onto the object", (x, y, z_grasp), 1.5, settle=True),
            WP("close on the object", (x, y, z_grasp), 0.0, kind="close"),
            WP("lift", (x, y, z_travel), 3.0),
            WP(f"hold {hold:.0f} s", (x, y, z_travel), hold, kind="hold"),
        ] + look2 + [
            WP("carry to the place point", (px, py, z_travel), 3.5),
            WP("lower", (px, py, z_grasp), 3.0, settle=True),
            WP("release", (px, py, z_grasp), 1.5, grip=GRIP_OPEN),
            WP("clear of the table", (px, py, z_travel), 2.5),
        ]
    wps.append(WP("fold back to ready", ready, 3.5, kind="pose", q=np.array(READY)))
    if park == "rest":
        wps.append(WP("return to rest", (0.0, 0.0, 0.0), 3.0, kind="rest"))
    return wps


def validate(planner: AzPlanner, ws: Workspace, plan: Plan, q_start: np.ndarray, step: float = 0.02,
             wps: list[WP] | None = None, holding: bool = False,
             placed: bool = False) -> tuple[float, str, list[tuple[str, float]], float]:
    """Walk the waypoints exactly as `execute` will - IK on every 2 cm sub-step, then sweep each joint
    segment against the no-go cylinders. Returns the worst clearance, what it was against, the clearance
    of every leg, and the lowest point any part of the arm reaches."""
    planner.az = plan.az
    carry = carried_points(plan.obj, plan.z_grasp, grasp_ahead(planner, plan))
    q = np.array(q_start[:6])
    worst, who, legs, low = 1e9, "", [], 1e9
    keep = ws.target
    ws.aim(None if holding else plan.obj, None if holding else (plan.place if placed else plan.pick))
    try:
        return _walk(planner, ws, plan, q, step, wps, holding, carry, worst, who, legs, low)
    finally:
        ws.target = keep


def _walk(planner, ws, plan, q, step, wps, holding, carry, worst, who, legs, low):
    for wp in (plan.waypoints if wps is None else wps):
        if wp.kind == "close":
            holding = True
            ws.aim(None)
            continue
        if wp.kind in ("hold", "rescan"):
            continue
        c = carry if holding else None
        if wp.kind in ("rest", "pose"):
            # REST is the SDK's folded pose the arm boots and parks in, so it needs no table check
            q_to = np.array(REST if wp.kind == "rest" else wp.q, dtype=float)
            wp.qs = [q_to]
            gap, w, zmin = ws.scan(q, q_to, c)
            if wp.kind == "pose" and zmin < table_floor(planner.fk_pos(q)[2], planner.fk_pos(q_to)[2]):
                raise RuntimeError(f"'{wp.label}': the arm dips into the table ({zmin * 100:.1f} cm)")
            q = q_to
        else:
            p0, p1 = planner.fk_pos(q), np.array(wp.xyz)
            n = max(1, int(np.linalg.norm(p1 - p0) / step)) if wp.kind == "line" else 1
            gap, w = 1e9, ""
            path = []
            for i in range(1, n + 1):
                p = p0 + (p1 - p0) * i / n
                q_next, _, _ = planner.ik(p[0], p[1], p[2], q, plan.yaw, plan.tilt)
                path.append(q_next)
                g, ww, zmin = ws.scan(q, q_next, c)
                if zmin < table_floor(planner.fk_pos(q)[2], p[2]):
                    raise RuntimeError(f"'{wp.label}': the arm dips into the table ({zmin * 100:.1f} cm)")
                if g < gap:
                    gap, w = g, ww
                q = q_next
            wp.qs = path
        if wp.grip is not None and wp.grip >= GRIP_OPEN - 1e-6:
            if holding:
                ws.aim(plan.obj, plan.place)  # released: it stands at the place point from here on
            holding = False
        legs.append((wp.label, gap))
        low = min(low, zmin)
        if gap < worst:
            worst, who = gap, w
    return worst, who, legs, low


JAW_ACROSS: np.ndarray | None = None  # unit xy direction the jaw must close across (--jaw-across)


def pose_options(mode: str) -> list[tuple[float, float, float | None]]:
    """(yaw, tilt, azimuth offset) candidates. For a top grasp the approach azimuth is redundant with the
    wrist yaw (the wrist points down), so the yaw is spun instead - half a turn, the jaw being symmetric -
    over the small outward tilts `pick_bottle` also needs, because a strictly vertical wrist runs into the
    wrist-pitch limit close to the base. A side grasp instead needs the direction the fingers come in
    from, which is searched around the radial one."""
    if mode == "top":
        return [(float(np.pi + a), float(t), None)
                for t in (0.0, 0.25, 0.45) for a in np.linspace(0, np.pi, 12, endpoint=False)]
    return [(np.pi, SIDE_TILT, float(d)) for d in (0.0, 0.35, -0.35, 0.7, -0.7, 1.05, -1.05, 1.4, -1.4)]


def solve_grasp(planner: AzPlanner, ws: Workspace, obj: Obj, mode: str, pick: tuple[float, float],
                z_grasp: float, z_travel: float) -> tuple[float, float, float | None]:
    """Choose the wrist yaw and approach azimuth with the most room, subject to IK (step 2)."""
    return grasp_candidates(planner, ws, obj, mode, pick, z_grasp, z_travel)[0]


def grasp_candidates(planner: AzPlanner, ws: Workspace, obj: Obj, mode: str, pick: tuple[float, float],
                     z_grasp: float, z_travel: float) -> list[tuple[float, float, float | None]]:
    """Every (yaw, tilt, azimuth) whose end points pass IK and the clearance pre-filter, best first. End
    points reaching is not the whole leg reaching (IK can lose its branch mid-slide), so callers validate
    in this order and take the first whose full trajectory passes."""
    radial = float(np.arctan2(pick[1], pick[0]))
    found = []
    for yaw, tilt, daz in pose_options(mode):
        az = None if daz is None else radial + daz
        jd = planner.jaw_dir(pick, yaw, tilt, az)[:2]
        jd = jd / (np.linalg.norm(jd) + 1e-9)
        if mode == "top" and JAW_ACROSS is not None and abs(float(jd @ JAW_ACROSS)) > np.sin(np.radians(12)):
            continue  # the jaw would close along the handle, landing the fingertips on it
        probes = [np.array(pick) + s * (obj.radius + JAW_GAP + 0.015) * jd for s in (1, -1)]  # the two tips
        if mode == "side":  # ... and the lane the fingers travel down
            ad = planner.approach_dir(pick, yaw, tilt, az)[:2]
            ad = ad / (np.linalg.norm(ad) + 1e-9)
            probes += [np.array(pick) - t * ad for t in np.linspace(0.02, SIDE_APPROACH, 5)]
        score = min(free_gap(p, ws.obstacles)[0] for p in probes)
        if score < 0:
            continue
        planner.az = az
        try:
            planner.ik(pick[0], pick[1], z_travel, REST, yaw, tilt)
            planner.ik(pick[0], pick[1], z_grasp, REST, yaw, tilt)
            if mode == "side":
                pre = np.array([pick[0], pick[1], z_grasp]) - SIDE_APPROACH * planner.approach_dir(pick, yaw, tilt, az)
                planner.ik(pre[0], pre[1], pre[2], REST, yaw, tilt)
        except RuntimeError:
            continue
        # approaches at or beyond GOOD_AZ first (see GOOD_AZ), then most room, then the truest vertical /
        # least contorted wrist
        a_deg = float(np.degrees(radial if az is None else az))
        key = (mode != "side" or a_deg <= GOOD_AZ, round(score, 3), -tilt, -abs(daz or 0.0))
        found.append((key, yaw, tilt, az))
    if not found:
        raise RuntimeError(f"no reachable, collision-free {mode} grasp")
    found.sort(key=lambda c: c[0], reverse=True)
    return [(y, t, a) for _, y, t, a in found]


def choose_place(planner: AzPlanner, ws: Workspace, obj: Obj, pick: tuple[float, float], yaw: float,
                 tilt: float, az: float | None, z_grasp: float, z_travel: float, dist: float,
                 forced_deg: float | None) -> tuple[float, float]:
    """A free spot `dist` from the pick point: the object's new footprint must clear every other object
    and both heights must be reachable there (step 4)."""
    planner.az = az
    cands = [np.radians(forced_deg)] if forced_deg is not None else list(np.linspace(0, 2 * np.pi, 24, endpoint=False))
    scored = []
    for a in cands:
        p = np.array(pick) + dist * np.array([np.cos(a), np.sin(a)])
        r = float(np.hypot(*p))
        if not (REACH[0] <= r <= REACH[1]):
            continue
        zone = load_zone()
        if zone is not None and zone_margin(p, zone) < obj.radius + 0.005:
            continue  # the whole footprint must land inside the zone
        gap, _ = free_gap(p, ws.obstacles, obj.radius)
        if gap < 0 and forced_deg is None:
            continue
        try:
            planner.ik(p[0], p[1], z_travel, REST, yaw, tilt)
            planner.ik(p[0], p[1], z_grasp, REST, yaw, tilt)
        except RuntimeError:
            continue
        # tie-break towards the spot at the same distance from the base as the pick: the arm keeps
        # roughly the same configuration, so the FK offset the visual check measured still holds there
        scored.append((round(gap, 3), -abs(r - float(np.hypot(*pick))), tuple(float(v) for v in p)))
    if not scored:
        raise RuntimeError(f"no free spot {dist * 100:.0f} cm from the pick point (try --place-dir)")
    scored.sort(reverse=True)
    return scored[0][2]


def servo_z(plan: Plan) -> float:
    """Fingertip height of the visual check: the grasp height itself. The FK offset it measures is mostly
    sag, which grows towards the table - checked at 16 cm it read 0.2 cm and the fingers closed ~2.5 cm off
    the bottle; checked at 7 cm a grasp still pushed the bottle 1.8 cm. At grasp height the check measures
    the sag the grasp will actually have.

    Reverted to grasp height + 5 cm: the two picks checked at grasp height (with correct three-camera positions)
    both missed, while the 7 cm check had completed most of its picks whenever the object's position was right."""
    return plan.z_grasp + 0.05


CHECK_SPARE = 0.01  # m: the check pose and its vertical drop/rise must clear by this much. The correction moves
# them by up to ~1 cm, and a check spot that cleared by exactly 0.0 cm made a 3 mm correction abort the run.


def check_point(plan: Plan) -> np.ndarray:
    """Where the visual check puts the fingertips (xy), as `visual_correct_3d` computes it."""
    if plan.servo_dir is not None:
        d = np.array(plan.servo_dir)
    else:
        d = load_cameras()[DET].C[:2] - np.array(plan.pick)
    return np.array(plan.pick) + plan.servo_dist * d / np.linalg.norm(d)


def drop_is_clear(planner: AzPlanner, ws: Workspace, plan: Plan, xy, z_top: float, z_bot: float) -> float:
    """Worst clearance (obstacles and the target's tolerance) of a vertical line at xy from z_top down to
    z_bot, walked the way `move_line` walks it."""
    q = planner.ik(xy[0], xy[1], z_top, REST, plan.yaw, plan.tilt)[0]
    worst = 1e9
    for z in np.linspace(z_top, z_bot, max(2, int((z_top - z_bot) / 0.02) + 1))[1:]:
        q_next = planner.ik(xy[0], xy[1], z, q, plan.yaw, plan.tilt)[0]
        worst = min(worst, ws.scan(q, q_next)[0])
        q = q_next
    return worst


def check_servo_point(planner: AzPlanner, ws: Workspace, plan: Plan) -> None:
    """Find where the visual check can happen: the fingertips at `servo_z`, 8-16 cm from the pick point,
    towards the camera first and then turning up to 90 deg either way. A spot qualifies only if it is
    reachable, clear of every other object, keeps the whole arm outside the target's tolerance, and puts
    the tips well inside the detection camera's frame. With no such spot the plan is marked unchecked and
    `execute` refuses to run it: open-loop, the arm's ~2.5 cm sag offset makes the fingers miss."""
    cam = load_cameras()[DET]
    w, h = cam.size
    to_cam = cam.C[:2] - np.array(plan.pick)
    base = float(np.arctan2(to_cam[1], to_cam[0]))
    planner.az = plan.az
    keep = ws.target
    ws.aim(plan.obj, plan.pick)
    why = "no candidate"
    try:
        for turn in (0, 30, -30, 60, -60, 90, -90):
            d = np.array([np.cos(base + np.radians(turn)), np.sin(base + np.radians(turn))])
            for dist in (SERVO_OFFSET, 0.10, 0.12, 0.14, 0.16):
                chk = np.array(plan.pick) + dist * d
                gap, who = free_gap(chk, ws.obstacles)
                if gap < 0:
                    why = f"the check point is {abs(gap) * 100:.1f} cm inside {who}"
                    continue
                u, v = cam.project(np.array([[chk[0], chk[1], servo_z(plan)]]))[0]
                if not (60 <= u <= w - 60 and 60 <= v <= h - 60):
                    why = "the check point is outside the camera's frame"
                    continue
                try:
                    q, _, _ = planner.ik(chk[0], chk[1], servo_z(plan), REST, plan.yaw, plan.tilt)
                except RuntimeError as e:
                    why = str(e)
                    continue
                tg = ws.target_gap(ws.points(q))
                if tg < CHECK_SPARE:
                    why = f"the arm is only {tg * 100:+.1f} cm outside the target's tolerance"
                    continue
                try:  # and the way in (and out): straight down from travel height onto the check point
                    drop = drop_is_clear(planner, ws, plan, chk, plan.z_travel, servo_z(plan))
                except RuntimeError as e:
                    why = f"the drop onto the check point: {e}"
                    continue
                if drop < CHECK_SPARE:
                    why = f"the drop onto the check point clears things by only {drop * 100:+.1f} cm"
                    continue
                plan.servo_dist, plan.servo_dir = dist, (float(d[0]), float(d[1]))
                return
        plan.servo_ok, plan.servo_note = False, why
    finally:
        ws.target = keep


def make_plan(objs: list[Obj], index: int, args: "Args", planner: AzPlanner) -> Plan:
    obj = objs[index]
    obstacles = [o for i, o in enumerate(objs) if i != index]
    if not (REACH[0] <= obj.reach <= REACH[1]):
        raise RuntimeError(f"{obj.name} is {obj.reach:.2f} m from the base: outside the safe reach {REACH}")
    mode, reason = decide_grasp(obj)
    if args.grasp != "auto":
        mode, reason = args.grasp, f"forced by --grasp {args.grasp}"
    ws = Workspace(planner, obstacles, args.margin)
    modes = [mode] if args.grasp != "auto" else [mode] + [m for m in ("top", "side") if m != mode]
    tried: list[str] = []
    fallback = None  # a plan that passes as detected but not with the pick shifted by a centimetre
    for m in modes:
        z0 = args.z_grasp if args.z_grasp is not None else grasp_height(m, obj)
        # a side grasp at the very bottom may leave the housing too close to a tall body: try a little higher
        zs = [z0] if (args.z_grasp is not None or m != "side") else [z0, z0 + 0.015, z0 + 0.03]
        for z_grasp in zs:
            z_travel = travel_height(obj, obstacles, z_grasp)
            try:
                cands = grasp_candidates(planner, ws, obj, m, obj.xy, z_grasp, z_travel)
            except RuntimeError as e:
                tried.append(f"{m}: {e}")
                continue
            last = ""
            for yaw, tilt, az in cands:
                try:
                    place_dir = args.place_dir
                    if args.shuttle and place_dir is None:  # repeated runs go back and forth between two spots
                        zone = load_zone()
                        if zone is not None:  # towards the zone's middle; from the middle, back out the way it came
                            to_mid = zone.mean(axis=0) - np.array(obj.xy)
                            ang = float(np.degrees(np.arctan2(to_mid[1], to_mid[0])))
                            place_dir = ang if np.linalg.norm(to_mid) > args.place_dist / 2 else ang + 180.0
                        else:
                            place_dir = 0.0 if obj.reach < SHUTTLE_R else 180.0
                    try:
                        place = choose_place(planner, ws, obj, obj.xy, yaw, tilt, az, z_grasp, z_travel,
                                             args.place_dist, place_dir)
                    except RuntimeError:
                        if place_dir is None or args.place_dir is not None:
                            raise
                        # the shuttle's preferred direction is blocked: any free direction will do
                        place = choose_place(planner, ws, obj, obj.xy, yaw, tilt, az, z_grasp, z_travel,
                                             args.place_dist, None)
                    gap_side = TALL_JAW_GAP if safe_height(obj) > TALL_H else JAW_GAP
                    open_grip = float(min(GRIP_OPEN, (obj.diameter + 2 * gap_side) / JAW_STROKE))
                    why = reason if m == mode else f"{reason} -- BUT that failed ({tried[-1]}), fell back to '{m}'"
                    plan = Plan(obj, obstacles, m, why, yaw, tilt, az, obj.xy, place, z_grasp, z_travel, open_grip)
                    pause = args.pause if args.rescan else 0.0
                    for park in ("rest", "ready"):
                        plan.waypoints = build_waypoints(planner, obj, m, yaw, tilt, az, obj.xy, place, z_grasp,
                                                         z_travel, open_grip, args.hold, pause, park)
                        gap, who, legs, low = validate(planner, ws, plan, REST)
                        # something parked against the base only blocks the way home: end upright at READY instead
                        if gap >= 0 or park == "ready" or [l for l, g in legs if g < 0] != ["return to rest"]:
                            break
                        print("  the way back to the folded rest pose is blocked; the run will park at READY instead")
                    if gap < 0:
                        leg = min(legs, key=lambda t: t[1])[0]
                        raise RuntimeError(f"'{leg}' passes {abs(gap) * 100:.1f} cm inside {who}'s no-go cylinder"
                                           + (f" (the arm already stands that close to {who} at rest; move it)"
                                              if leg == "unfold to ready" else ""))
                    plan.clearance, plan.tight_at, plan.legs, plan.low_point = gap, who, legs, low
                    plan.margin = args.margin
                    if args.servo:
                        check_servo_point(planner, ws, plan)
                    weak = fragile(plan, planner, ws, args)
                    if not weak:
                        return plan
                    fallback = fallback or (plan, weak)
                    last = f"fragile ({weak})"
                except RuntimeError as e:
                    last = str(e)
            tried.append(f"{m} at {z_grasp * 100:.1f} cm: all {len(cands)} approach direction(s) failed, the last: {last}")
    if fallback is not None:
        plan, weak = fallback
        print(f"  note: no approach survives a 1 cm shift of the pick ({weak}); using the best one - a visual "
              "correction may still find it unreachable and abort")
        return plan
    raise RuntimeError("no safe pick-and-place found -> " + " | ".join(tried))


ROBUST_SHIFT = 0.01  # m: the visual check moves the pick by up to about this much


def fragile(plan: "Plan", planner: AzPlanner, ws: "Workspace", args: "Args") -> str:
    """Why this plan would break if the pick moved by ROBUST_SHIFT in any direction, or "" if it would not.
    A plan valid only at the camera's estimate is a knife-edge: a 0.5 cm visual correction pushed one's
    slide-in out of IK reach and the next-best approach 0.1 cm inside the target (zone batch 2, run 1)."""
    pause = args.pause if args.rescan else 0.0
    for dx, dy in ((ROBUST_SHIFT, 0), (-ROBUST_SHIFT, 0), (0, ROBUST_SHIFT), (0, -ROBUST_SHIFT)):
        s = np.array([dx, dy])
        alt = rebuild(plan, planner, tuple(np.array(plan.pick) + s), tuple(np.array(plan.place) + s), args.hold, pause)
        try:
            gap, who, _, _ = validate(planner, ws, alt, REST)
        except RuntimeError as e:
            return f"shifted {dx * 100:+.0f},{dy * 100:+.0f} cm: {e}"
        if gap < 0:
            return f"shifted {dx * 100:+.0f},{dy * 100:+.0f} cm: {abs(gap) * 100:.1f} cm inside {who}"
    planner.az = plan.az
    return ""


def rebuild(plan: Plan, planner: AzPlanner, pick: tuple[float, float], place: tuple[float, float],
            hold: float, pause: float) -> Plan:
    """Same grasp mode, wrist pose and heights, new pick and/or place point. The waypoint list keeps its
    shape, so a leg index stays valid across a rebuild mid-run."""
    p = Plan(plan.obj, plan.obstacles, plan.mode, plan.reason, plan.yaw, plan.tilt, plan.az, pick, place,
             plan.z_grasp, plan.z_travel, plan.open_grip, [], plan.clearance, plan.tight_at, plan.legs,
             plan.low_point, plan.margin, plan.servo_ok, plan.servo_note, plan.servo_dist, [], plan.servo_dir)
    p.waypoints = build_waypoints(planner, plan.obj, plan.mode, plan.yaw, plan.tilt, plan.az, pick, place,
                                  plan.z_grasp, plan.z_travel, plan.open_grip, hold, pause)
    return p


def replan(plan: Plan, planner: AzPlanner, pick: tuple[float, float], hold: float, pause: float) -> Plan:
    """Shift the plan onto a corrected pick point (after the visual check); the place point rides along
    on the same offset, which was already validated against the obstacles."""
    off = np.array(plan.place) - np.array(plan.pick)
    return rebuild(plan, planner, pick, tuple(float(v) for v in np.array(pick) + off), hold, pause)


def validate_from_check(planner: AzPlanner, ws: Workspace, plan: Plan, q: np.ndarray):
    """What `execute` actually does after the visual check: straight up out of the check pose, over to
    above the (corrected) pick point, then waypoints[2:]. Validating the whole plan from the check pose
    would walk the unfold/approach legs from beside the object - legs that never run - through it."""
    here = planner.fk_pos(q)
    lead = [WP("up out of the visual check", (float(here[0]), float(here[1]), plan.z_travel), 1.5),
            WP("over to above the object", (plan.pick[0], plan.pick[1], plan.z_travel), 2.0)]
    out = validate(planner, ws, plan, q, wps=lead + plan.waypoints[2:])
    plan.lead = lead  # execute follows these joints out of the check pose
    return out


def fit_corrected(plan: Plan, planner: AzPlanner, ws: Workspace, cmd: tuple[float, float], args: "Args",
                  pause: float, q: np.ndarray) -> tuple[Plan, float, str, list]:
    """Move the plan onto the command point the visual check produced. The same wrist pose first; if the
    arm cannot reach it that way (a large FK correction pulls the pick towards the base, where a 60 deg
    side grasp runs out of IK) or it is no longer clear, search the approach directions again there.
    Returns the plan and its validation; raises if nothing reachable and clear is left."""
    off = np.array(plan.place) - np.array(plan.pick)
    tried = []
    try:
        p = replan(plan, planner, cmd, args.hold, pause)
        gap, who, legs, _ = validate_from_check(planner, ws, p, q)
        if gap >= 0:
            p.legs = legs
            return p, gap, who, legs
        tried.append(f"same approach comes {abs(gap) * 100:.1f} cm inside {who}")
    except RuntimeError as e:
        tried.append(f"same approach: {e}")
    print(f"   {tried[-1]}; searching the approach directions again at {np.round(cmd, 3)}")
    last = ""
    for yaw, tilt, az in grasp_candidates(planner, ws, plan.obj, plan.mode, cmd, plan.z_grasp, plan.z_travel):
        try:
            return _fit_with(plan, planner, ws, cmd, off, args, pause, q, yaw, tilt, az)
        except RuntimeError as e:
            last = str(e)
    raise RuntimeError(f"no approach reaches the corrected pick point ({last})")


def _fit_with(plan: Plan, planner: AzPlanner, ws: Workspace, cmd, off, args: "Args", pause: float,
              q: np.ndarray, yaw: float, tilt: float, az: float | None) -> tuple[Plan, float, str, list]:
    base = Plan(plan.obj, plan.obstacles, plan.mode, plan.reason, yaw, tilt, az, cmd,
                tuple(float(v) for v in np.array(cmd) + off), plan.z_grasp, plan.z_travel, plan.open_grip,
                [], plan.clearance, plan.tight_at, plan.legs, plan.low_point, plan.margin, plan.servo_ok,
                plan.servo_note, plan.servo_dist, [], plan.servo_dir)
    p = rebuild(base, planner, base.pick, base.place, args.hold, pause)
    gap, who, legs, _ = validate_from_check(planner, ws, p, q)
    if gap < 0:
        raise RuntimeError(f"best new approach still comes {abs(gap) * 100:.1f} cm inside {who}")
    az_txt = "radial" if az is None else f"{np.degrees(az):.0f} deg"
    print(f"   new approach: azimuth {az_txt}, wrist yaw {np.degrees(yaw):.0f} deg, tilt {np.degrees(tilt):.0f} deg")
    p.legs = legs
    return p, gap, who, legs


# ==================================================================================== reporting
def print_plan(plan: Plan) -> None:
    print("\nstep 1  identify")
    print(f"  target   {plan.obj.describe()}")
    for ob in plan.obstacles:
        print(f"  obstacle {ob.describe()}")
    print("\nstep 3  grasp decision")
    print(f"  {'VERTICAL (top-down)' if plan.mode == 'top' else 'HORIZONTAL (side)'}: {plan.reason}")
    print(f"  fingertips close at z {plan.z_grasp * 100:.1f} cm; jaw pre-opens to "
          f"{plan.open_grip * JAW_STROKE * 100:.1f} cm for a {plan.obj.diameter * 100:.1f} cm object")
    print("\nstep 2  trajectory")
    print(f"  travel height {plan.z_travel * 100:.0f} cm, wrist yaw {np.degrees(plan.yaw):.0f} deg, tilt "
          f"{np.degrees(plan.tilt):.0f} deg, approach azimuth "
          f"{'radial' if plan.az is None else f'{np.degrees(plan.az):.0f} deg'}")
    moves = [w for w in plan.waypoints if w.kind not in ("hold", "close", "rescan")]
    for (label, gap), wp in zip(plan.legs, moves):
        print(f"  {label:<28} -> ({wp.xyz[0]:+.3f}, {wp.xyz[1]:+.3f}, {wp.xyz[2]:.3f})   clearance {fmt_gap(gap)}")
    print(f"  tightest point: {fmt_gap(plan.clearance)}"
          + (f" from {plan.tight_at} (the {plan.margin * 100:.1f} cm margin is already subtracted)"
             if plan.tight_at else " to stay clear of")
          + f"; the lowest part of the arm on the whole path is {plan.low_point * 100:.1f} cm above the table")
    if not plan.servo_ok:
        print(f"  visual check DISABLED ({plan.servo_note}): the run is refused unless --open-loop")
    print("\nstep 4  pick and place")
    d = np.array(plan.place) - np.array(plan.pick)
    print(f"  pick ({plan.pick[0]:+.3f}, {plan.pick[1]:+.3f}) -> hold -> place ({plan.place[0]:+.3f}, "
          f"{plan.place[1]:+.3f}): {np.linalg.norm(d) * 100:.1f} cm at {np.degrees(np.arctan2(d[1], d[0])):.0f} deg\n")


def draw_plan(frame: np.ndarray, cam: CameraModel, plan: Plan, out: Path) -> None:
    img = frame.copy()
    th = np.linspace(0, 2 * np.pi, 41)
    for o, col in [(plan.obj, (0, 255, 0))] + [(ob, (0, 165, 255)) for ob in plan.obstacles]:
        for z in (0.0, o.height):
            ring = np.array([[o.xy[0] + o.radius * np.cos(a), o.xy[1] + o.radius * np.sin(a), z] for a in th])
            cv2.polylines(img, [cam.project(ring).astype(np.int32)], True, col, 1)
        u, v = cam.project(np.array([[o.xy[0], o.xy[1], o.height]]))[0]
        cv2.putText(img, f"{o.name} {o.diameter * 100:.0f}x{o.height * 100:.0f}cm", (int(u) - 60, int(v) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    path = np.array([w.xyz for w in plan.waypoints if w.kind in ("line", "joint", "pose")])
    px = cam.project(path).astype(np.int32)
    cv2.polylines(img, [px], False, (255, 0, 255), 2)
    for p in px:
        cv2.drawMarker(img, tuple(p), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 2)
    cv2.drawMarker(img, tuple(cam.project(np.array([[*plan.pick, 0.0]]))[0].astype(int)), (0, 0, 255),
                   cv2.MARKER_CROSS, 24, 2)
    cv2.drawMarker(img, tuple(cam.project(np.array([[*plan.place, 0.0]]))[0].astype(int)), (255, 255, 0),
                   cv2.MARKER_STAR, 24, 2)
    cv2.putText(img, f"{'VERTICAL' if plan.mode == 'top' else 'HORIZONTAL'} pick -> place "
                     f"{np.linalg.norm(np.array(plan.place) - np.array(plan.pick)) * 100:.0f} cm away",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(out), img)


# ============================================================== mid-run re-analysis of the scene
MATCH_R = 0.06  # m: a fresh detection this close to a known object is that same object


def merge_objects(known: list[Obj], fresh: list[Obj], mask: np.ndarray,
                  cam: CameraModel) -> tuple[list[Obj], list[str]]:
    """Fresh detections win; an object that has vanished only because the arm is standing in front of
    it keeps its last known position, because it is still there to be knocked over."""
    out, notes = list(fresh), []
    for f in fresh:
        d = min((float(np.hypot(*(np.array(f.xy) - np.array(k.xy)))) for k in known), default=9.0)
        if d > MATCH_R:
            notes.append(f"NEW object at {np.round(f.xy, 3)} ({f.diameter * 100:.0f} x {f.height * 100:.0f} cm)")
        elif d > 0.012:
            notes.append(f"an object moved {d * 100:.1f} cm (now at {np.round(f.xy, 3)})")
    for k in known:
        if min((float(np.hypot(*(np.array(f.xy) - np.array(k.xy)))) for f in fresh), default=9.0) <= MATCH_R:
            continue
        u, v = cam.project(np.array([[k.xy[0], k.xy[1], 0.0]]))[0]
        iu, iv = int(round(u)), int(round(v))
        hidden = 0 <= iv < mask.shape[0] and 0 <= iu < mask.shape[1] and mask[iv, iu] > 0
        if hidden:
            out.append(Obj(f"{k.name} (hidden)", k.xy, k.diameter, k.height, k.color))
            notes.append(f"{k.name} is behind the arm: keeping its last known position")
        else:
            notes.append(f"{k.name} is no longer on the table")
    return out, notes


def rescan(plan: Plan, planner: AzPlanner, ws: Workspace, arm: Arm, cam: LiveCamera, cmodel: CameraModel,
           args: "Args", idx: int, holding: bool) -> tuple[Plan, bool]:
    """Stop, look at the table again with the arm masked out of the frame, update the obstacles and
    re-check everything that is left of the trajectory. If the way to the place point is now blocked,
    a new place point is chosen. Returns the plan to carry on with, and whether it is safe to do so.

    The target's own position is NOT re-estimated here: the gripper is hovering right over it, so most
    of it is behind the mask, and the visual check already put the command where the fingers must go."""
    print(f"-> pausing {args.pause:.1f} s and re-reading the table")
    time.sleep(args.pause)
    q = arm.q()
    tip = planner.fk_pos(q)
    # every calibrated camera (the fused workspace); the detection camera alone when only it is open
    views = {k: c for k, c in FUSED_CAMS.items() if k in FUSED_MODELS} or {DET: cam}
    models = FUSED_MODELS or {DET: cmodel}
    frames = {k: c.latest() for k, c in views.items()}
    masks = {}
    for k, f in frames.items():
        m = arm_mask(ws, models[k], q, f.shape[:2])
        if holding:
            # the object hanging in the gripper is not on the table: mask it and ignore anything read under
            # the fingertips (a side grasp holds it off-axis, where it read as a NEW 5 x 11 cm obstacle)
            px = models[k].project(carried_points(plan.obj, plan.z_grasp, grasp_ahead(planner, plan))(tip))
            px = px[np.isfinite(px).all(axis=1) & (np.abs(px) < 1e4).all(axis=1)]
            if len(px) >= 3:
                c = np.zeros(f.shape[:2], np.uint8)
                cv2.fillConvexPoly(c, cv2.convexHull(px.astype(np.int32)), 255)
                m |= cv2.dilate(c, np.ones((41, 41), np.uint8))
        masks[k] = m
    frame, mask = frames[DET], masks[DET]
    near = [plan.pick] + ([(float(tip[0]), float(tip[1]))] if holding else [])
    found = survey_multi(frames, models, args.min_area, ignore=masks) if len(frames) > 1 else \
        survey(frame, cmodel, args.min_area, ignore=mask)
    fresh = [o for o in found
             if min(float(np.hypot(*(np.array(o.xy) - np.array(p)))) for p in near) > MATCH_R + plan.obj.radius]
    dbg = frame.copy()
    dbg[mask > 0] = (dbg[mask > 0] * 0.35).astype(np.uint8)
    for o in fresh:
        u, v = cmodel.project(np.array([[o.xy[0], o.xy[1], 0.0]]))[0]
        cv2.circle(dbg, (int(u), int(v)), 8, (0, 165, 255), 2)
    cv2.imwrite(str(CAPTURES / f"rescan_{idx}.jpg"), dbg)
    obstacles, notes = merge_objects(plan.obstacles, fresh, mask, cmodel)
    for n in notes:
        print(f"   {n}")
    if not notes:
        print(f"   scene unchanged: {len(obstacles)} object(s) to stay clear of")
    ws.obstacles = obstacles
    plan.obstacles = obstacles
    try:
        return _recheck(plan, planner, ws, q, args, idx, holding, obstacles)
    except RuntimeError as e:  # an unreachable leg is as unsafe as a blocked one
        print(f"   {e}")
        return plan, False


def _recheck(plan: Plan, planner: AzPlanner, ws: Workspace, q: np.ndarray, args: "Args", idx: int,
             holding: bool, obstacles: list[Obj]) -> tuple[Plan, bool]:
    gap, who, _, _ = validate(planner, ws, plan, q, wps=plan.waypoints[idx + 1 :], holding=holding)
    if gap >= 0:
        print(f"   the rest of the trajectory is clear ({fmt_gap(gap)})")
        return plan, True
    print(f"   the rest of the trajectory would come {abs(gap) * 100:.1f} cm inside {who}")
    try:
        place = choose_place(planner, ws, plan.obj, plan.pick, plan.yaw, plan.tilt, plan.az, plan.z_grasp,
                             plan.z_travel, args.place_dist, args.place_dir)
    except RuntimeError as e:
        print(f"   {e}")
        return plan, False
    alt = rebuild(plan, planner, plan.pick, place, args.hold, args.pause if args.rescan else 0.0)
    alt.obstacles = obstacles
    gap, who, _, _ = validate(planner, ws, alt, q, wps=alt.waypoints[idx + 1 :], holding=holding)
    if gap < 0:
        print(f"   still blocked by {who} after re-picking the place point")
        return plan, False
    d = np.array(place) - np.array(plan.pick)
    print(f"   new place point {np.round(place, 3)} ({np.linalg.norm(d) * 100:.1f} cm at "
          f"{np.degrees(np.arctan2(d[1], d[0])):.0f} deg), clearance {fmt_gap(gap)}")
    return alt, True


# the cameras of the fused workspace while a run is going (stage_run fills these)
FUSED_CAMS: dict = {}
FUSED_MODELS: dict = {}


# ============================================================================== live workspace map
def plan_path(planner: AzPlanner, plan: Plan) -> list[list[float]]:
    """The fingertip polyline the plan will follow, for the map."""
    out: list[list[float]] = []
    for wp in plan.waypoints:
        if wp.kind in ("rescan", "hold", "close"):
            continue
        xyz = planner.fk_pos(REST if wp.kind == "rest" else wp.q) if wp.kind in ("rest", "pose") else wp.xyz
        pt = [round(float(v), 4) for v in xyz]
        if not out or pt != out[-1]:
            out.append(pt)
    return out


def show_plan(live: LiveState | None, planner: AzPlanner, plan: Plan) -> None:
    if live is None:
        return
    keep = planner.az
    path = plan_path(planner, plan)
    planner.az = keep
    live.set(mode=plan.mode, reason=plan.reason, z_grasp=round(plan.z_grasp, 4),
             pick=[round(float(v), 4) for v in plan.pick], place=[round(float(v), 4) for v in plan.place],
             steps=[w.label for w in plan.waypoints], path=path,
             objects=[{"name": plan.obj.name, "axis": list(plan.obj.xy), "diameter": plan.obj.diameter,
                       "height": plan.obj.height, "role": "target"}]
             + [{"name": o.name, "axis": list(o.xy), "diameter": o.diameter, "height": o.height,
                 "role": "obstacle"} for o in plan.obstacles])


def view_gate(plan: Plan, planner: AzPlanner, ws: Workspace, q_start: np.ndarray, frames: dict | None,
              label: str, wps: list | None = None) -> bool:
    """The fused-view check: sweep every arm vertex along the exact joint paths `validate` stored on the
    waypoints (wp.qs) and test each point against the fused objects - obstacles with the margin, the target
    with its tolerance (fingertips excepted, and only while it stands on the table). Renders TOP (the
    multi-camera homography), SIDE and ISOMETRIC views of it to captures/fused_views.jpg and returns
    whether every point is clear. `execute` refuses to move on False."""
    from fused import TopView, render_views

    walk = plan.waypoints if wps is None else wps
    carry = carried_points(plan.obj, plan.z_grasp, grasp_ahead(planner, plan))
    q = np.array(q_start[:6], dtype=float)
    holding, placed = False, False
    swept, bad = [], []
    obst = [(np.array(o.xy), o.radius + ws.margin, o.height + 0.01) for o in plan.obstacles]
    th = safe_height(plan.obj)
    tol = TALL_TOL if th > TALL_H else TARGET_TOL
    for wp in walk:
        if wp.kind == "close":
            holding = True
            continue
        for q_next in (wp.qs or []):
            q_next = np.asarray(q_next, dtype=float)
            for a in np.linspace(0.25, 1.0, 4):
                qq = (1 - a) * q + a * q_next
                pts = ws.points(qq)
                tip = ws.is_tip
                b = np.zeros(len(pts), bool)
                for c, r, h in obst:
                    b |= (pts[:, 2] < h) & (np.linalg.norm(pts[:, :2] - c, axis=1) < r)
                if not holding:  # the target standing on the table (pick point, or place point once released)
                    c = np.array(plan.place if placed else plan.pick)
                    b |= ~tip & (pts[:, 2] < th + 0.01) & (np.linalg.norm(pts[:, :2] - c, axis=1) < plan.obj.radius + tol)
                if holding:
                    cp = carry(planner.fk_pos(qq))
                    cb = np.zeros(len(cp), bool)
                    for c, r, h in obst:
                        cb |= (cp[:, 2] < h) & (np.linalg.norm(cp[:, :2] - c, axis=1) < r)
                    pts, b = np.vstack([pts, cp]), np.r_[b, cb]
                swept.append(pts[::3])
                bad.append(b[::3])
            q = q_next
        if wp.grip is not None and wp.grip >= GRIP_OPEN - 1e-6 and holding:
            holding, placed = False, True
    swept, bad = np.vstack(swept), np.concatenate(bad)
    ok = not bad.any()
    try:
        zone = load_zone()
        objs = [{"name": plan.obj.name, "axis": list(plan.obj.xy), "diameter": plan.obj.diameter,
                 "height": plan.obj.height, "role": "target"}] + \
               [{"name": o.name, "axis": list(o.xy), "diameter": o.diameter, "height": o.height, "role": "obstacle"}
                for o in plan.obstacles]
        models = FUSED_MODELS or load_cameras()
        tv = TopView(models, zone)
        top = tv.render(frames) if frames else np.full((tv.h, tv.w, 3), 235, np.uint8)
        keep = planner.az
        path = np.array(plan_path(planner, plan))
        planner.az = keep
        title = (f"{label}: " + ("every arm point clear of every object" if ok else
                 f"{int(bad.sum())} arm points inside a clearance - NOT RUNNING"))
        img = render_views(top, tv, zone, objs, path, swept, bad, title)
        cv2.imwrite(str(CAPTURES / "fused_views.jpg"), img)
    except Exception as e:  # the picture is for people; the verdict above does not depend on it
        print(f"   (could not draw the fused views: {e})")
    print(f"   fused-view check ({label}): " + ("clear" if ok else f"{int(bad.sum())} points inside a clearance") +
          " -> captures/fused_views.jpg")
    return ok


def safe_park(arm: Arm, planner: AzPlanner, yaw: float, tilt: float) -> None:
    """Park after an error when nothing is held: straight up to travel height first, keeping the wrist as
    the plan had it (a joint-space glide to REST from low beside an object sweeps the arm through it),
    then through READY to REST."""
    arm.set_grip(GRIP_OPEN, 1.0)
    try:
        q = np.array(arm.last_cmd[:6], dtype=float)
        here = planner.fk_pos(q)
        steps = max(1, int(np.ceil((Z_TRAVEL_MIN - here[2]) / 0.02)))
        for k in range(1, steps + 1) if here[2] < Z_TRAVEL_MIN else ():
            z = here[2] + (Z_TRAVEL_MIN - here[2]) * k / steps
            q, _, _ = planner.ik(here[0], here[1], z, q, yaw, tilt)
            arm.glide(q, 1.5 / steps)
    except RuntimeError as e:
        print(f"   (no straight lift from here: {e})")
    goto_joint(arm, planner, np.array(READY), 4.0)
    goto_joint(arm, planner, REST, 4.0)


def bail(arm: Arm, planner: AzPlanner, ws: Workspace, plan: Plan, holding: bool, reason: str) -> None:
    """Stop the workflow without leaving the object dangling: put it back where it came from (that spot
    is known to be clear, it is where the object was standing), then park."""
    print(f"!! {reason}: aborting")
    try:
        if holding:
            print("-> putting the object back where it was picked up")
            bx, by = np.array(plan.pick) - grasp_ahead(planner, plan)
            move_line(arm, planner, ws, (bx, by, plan.z_travel), plan.yaw, plan.tilt, 3.0)
            move_line(arm, planner, ws, (bx, by, plan.z_grasp), plan.yaw, plan.tilt, 3.0)
            settle(arm, planner, bx, by, plan.z_grasp, plan.yaw, plan.tilt)
        arm.set_grip(GRIP_OPEN, 1.5)
        if holding:
            move_line(arm, planner, ws, (bx, by, plan.z_travel), plan.yaw, plan.tilt, 2.5)
    except RuntimeError as e:
        print(f"   could not set it down cleanly ({e}); opening the gripper where it stands")
        arm.set_grip(GRIP_OPEN, 1.5)
    safe_park(arm, planner, plan.yaw, plan.tilt)  # straight up first, never a joint-space swing past the object


# ==================================================================================== execution
def grip_on_object(arm: Arm, args: "Args") -> bool:
    """Close until the fingers stall on the object. True if something is actually held."""
    g = close_until_contact(arm, squeeze=args.squeeze)
    holding = g > 0.06
    print(f"   gripper at {g:.2f} ({g * JAW_STROKE * 100:.1f} cm): "
          f"{'holding the object' if holding else 'CLOSED ON NOTHING'}")
    return holding


def execute(plan: Plan, planner: AzPlanner, arm: Arm, cam: LiveCamera | None, cmodel: CameraModel,
            args: "Args", live: LiveState | None = None) -> bool:
    """Run the waypoints. Leg 0 is the joint move to the travel pose above the object; the visual check
    happens there and the rest of the plan is rebuilt around the corrected xy. The "look at the scene
    again" legs stop the arm and re-check everything that is still ahead of it."""
    planner.az = plan.az
    ws = Workspace(planner, plan.obstacles, args.margin)
    ws.aim(plan.obj, plan.pick)
    pause = args.pause if args.rescan else 0.0

    def step(i: int, label: str) -> None:
        print(f"-> {label}")
        if live is not None:
            live.set(phase=label, step=i)

    # the plan was validated from the folded rest pose; check it again from wherever the arm actually is
    gap, who, legs, _ = validate(planner, ws, plan, arm.q())
    print(f"from the arm's current pose: clearance {fmt_gap(gap)}" + (f" from {who}" if who else ""))
    if gap < 0:
        print(f"!! {min(legs, key=lambda t: t[1])[0]} would come {abs(gap) * 100:.1f} cm inside {who}: "
              "not moving")
        return False
    frames = {k: c.latest() for k, c in FUSED_CAMS.items()} if FUSED_CAMS else None
    if not view_gate(plan, planner, ws, arm.q(), frames, "planned trajectory"):
        print("!! the fused-view check found the arm inside a clearance: not moving")
        return False
    if args.servo and not plan.servo_ok and not args.open_loop:
        print(f"!! no safe spot for the visual check ({plan.servo_note}): not running open-loop - the arm's sag "
              "offset makes the fingers miss. Move the object, or pass --open-loop to accept that.")
        return False
    unfold, above = plan.waypoints[0], plan.waypoints[1]
    step(0, unfold.label)
    arm.set_grip(unfold.grip if unfold.grip is not None else GRIP_OPEN, 1.0)
    goto_joint(arm, planner, np.array(unfold.q), unfold.seconds)
    step(1, above.label)
    q = above.qs[-1] if above.qs else planner.ik(above.xyz[0], above.xyz[1], above.xyz[2], arm.q(), plan.yaw,
                                                  plan.tilt)[0]
    goto_joint(arm, planner, np.asarray(q, dtype=float), above.seconds)
    if args.servo and plan.servo_ok and cam is not None:
        step(1, "visual check next to the object")
        # get there over the top and then straight down: the check's own straight line from above the object
        # to the check point cut through a 10 cm bottle and knocked it off the zone (zone batch, run 2)
        chk = check_point(plan)
        move_line(arm, planner, ws, (chk[0], chk[1], plan.z_travel), plan.yaw, plan.tilt, 2.5)
        move_line(arm, planner, ws, (chk[0], chk[1], servo_z(plan)), plan.yaw, plan.tilt, 2.5)
        cmd = visual_correct_3d(arm, planner, cam, np.array(plan.pick), plan.yaw, plan.tilt, servo_z(plan),
                                cam_key=DET, offset=plan.servo_dist, direction=plan.servo_dir)
        try:
            plan, gap, who, legs = fit_corrected(plan, planner, ws, (float(cmd[0]), float(cmd[1])), args,
                                                 pause, arm.q())
        except RuntimeError as e:
            bail(arm, planner, ws, plan, False, f"no safe trajectory to the corrected pick point ({e})")
            return False
        planner.az = plan.az
        show_plan(live, planner, plan)
        print(f"   corrected plan: clearance {fmt_gap(gap)}" + (f" from {who}" if who else ""))
        frames = {k: c.latest() for k, c in FUSED_CAMS.items()} if FUSED_CAMS else None
        if not view_gate(plan, planner, ws, arm.q(), frames, "corrected trajectory", plan.lead + plan.waypoints[2:]):
            bail(arm, planner, ws, plan, False, "the corrected trajectory fails the fused-view check")
            return False
        plan.legs = legs
        ws.aim(plan.obj, plan.pick)
        # straight up out of the check pose first (a diagonal from there swings the tilted side-grasp
        # gripper through the table), then over - along exactly the joints fit_corrected validated
        for wp in plan.lead:
            move_line(arm, planner, ws, wp.xyz, plan.yaw, plan.tilt, wp.seconds, qs=wp.qs)
    i = 2
    try:
        return _run_legs(plan, planner, arm, cam, cmodel, args, live, ws, step, i)
    except RuntimeError as e:
        if not getattr(arm, "holding", False):
            raise
        bail(arm, planner, ws, plan, True, str(e))  # never open the gripper in the air
        return False


def _run_legs(plan: Plan, planner: AzPlanner, arm: Arm, cam: LiveCamera | None, cmodel: CameraModel,
              args: "Args", live: LiveState | None, ws: Workspace, step, i: int) -> bool:
    holding, carry = False, None
    arm.holding = False
    ws.aim(plan.obj, plan.pick)  # in the planner's frame the object stands at the commanded pick point
    while i < len(plan.waypoints):
        wp = plan.waypoints[i]
        if wp.kind == "rescan":
            if cam is None:
                print("-> (no camera: skipping the re-analysis)")
            else:
                if live is not None:
                    live.set(phase=wp.label, step=i)
                plan, ok = rescan(plan, planner, ws, arm, cam, cmodel, args, i, holding)
                planner.az = plan.az
                show_plan(live, planner, plan)
                if not ok:
                    bail(arm, planner, ws, plan, holding, "the scene changed and no safe trajectory is left")
                    return False
            i += 1
            continue
        step(i, wp.label)
        if wp.kind == "close":
            holding, carry = True, carried_points(plan.obj, plan.z_grasp, grasp_ahead(planner, plan))
            arm.holding = True
            ws.aim(None)
            if not grip_on_object(arm, args):
                arm.holding = False
                bail(arm, planner, ws, plan, False, "the gripper closed on nothing")
                return False
        elif wp.kind == "hold":
            time.sleep(wp.seconds)
        elif wp.kind in ("rest", "pose"):
            goto_joint(arm, planner, np.array(REST if wp.kind == "rest" else wp.q), wp.seconds)
        else:
            move_line(arm, planner, ws, wp.xyz, plan.yaw, plan.tilt, wp.seconds, carry, qs=wp.qs)
            if wp.settle:  # a refinement on top of a leg that already arrived: skip it if IK has no answer
                try:
                    pos = settle(arm, planner, wp.xyz[0], wp.xyz[1], wp.xyz[2], plan.yaw, plan.tilt)
                    print(f"   fingertips at {np.round(pos, 3)}")
                except RuntimeError as e:
                    print(f"   (sag compensation skipped: {e})")
            if wp.grip is not None:
                arm.set_grip(wp.grip, 1.5)
                if wp.grip >= GRIP_OPEN - 1e-6:
                    holding, carry = False, None
                    arm.holding = False
                    ws.aim(plan.obj, plan.place)
        i += 1
    print("done")
    return True


# ========================================================================================== CLI
@dataclass
class Args:
    scene: bool = False
    """Camera only: recreate the 3D workspace map (scene.json, workspace_map.html) from every calibrated camera."""
    zone: bool = False
    """Find the blue-taped white zone in the empty-zone background -> zone.json. Re-run after the tape or a
    camera moves (and re-run calib3d.py first if it was a camera)."""
    background: bool = False
    """Capture captures/bg_cam*.png with an EMPTY table; re-run whenever the lighting or a camera moves."""
    given: str | None = None
    """"x,y,diameter,height" (m): pick this object instead of a detected one - for a target the survey cannot
    see (clear plastic on the white sheet), measured by hand from the cameras. It becomes object 0; anything
    the survey does detect is kept as an obstacle. More "x,y,d,h" after ";" are extra obstacles (the head of
    a hammer whose handle is the target); detections within 6 cm of any given object are dropped."""
    given_only: bool = False
    """With --given: use only the given objects, dropping every detection (a lying hammer's handle is detected
    as a separate short cylinder)."""
    jaw_across: str | None = None
    """"dx,dy": a top grasp only takes wrist yaws whose jaw closes across this direction (a handle's axis), so
    the fingers straddle the handle instead of landing on it."""
    survey: bool = False
    plan: bool = False
    run: bool = False
    sim: bool = False
    object: int = 0
    """Index of the object to pick, as printed by --survey."""
    frames: str | None = None
    """Plan offline from this saved cam-0 image instead of grabbing a new one."""
    grasp: str = "side"
    """'side' (default: horizontal, fingers close around the bottom of the object), 'top' (vertical), or
    'auto' (the flatness/height rule, falling back to the other mode when one has no safe trajectory)."""
    place_dist: float = 0.05
    """Metres from the pick point to set the object back down."""
    place_dir: float | None = None
    """Force the place direction, degrees in the robot frame (default: the clearest one)."""
    shuttle: bool = False
    """For repeated runs: set the object down 5 cm outward (+x) when it is nearer the base than SHUTTLE_R and
    5 cm back (-x) otherwise, so it goes back and forth between two spots instead of walking off the table."""
    hold: float = 5.0
    """Seconds to hold the object in the air before placing it."""
    z_grasp: float | None = None
    """Override the fingertip height at closing (default: derived from the object's height)."""
    squeeze: float = 0.05
    """Extra closing after contact, in gripper units (~5 mm). 0.03 for rigid things, 0.08 for soft ones."""
    margin: float = MARGIN
    """Metres of clearance to keep from every object that is not the target."""
    min_area: float = 2000.0
    """Smallest blob (px) treated as an object."""
    servo: bool = True
    """Blink the fingertips next to the object and correct the arm's FK offset before grasping."""
    open_loop: bool = False
    """Run even when the visual check has no safe spot (the grasp then relies on the camera fit alone)."""
    rescan: bool = True
    """Stop mid-run and read the table again (before descending, and before carrying the object across):
    obstacles are updated, the rest of the trajectory is re-checked and the place point is moved if needed."""
    pause: float = 1.5
    """Seconds to hold still at each of those stops before grabbing the frame."""
    cam: int = 0
    cam2: int = 1
    channel: str = "can0"
    hz: float = 100.0
    record: str | None = "datasets/yam_pick_place"
    """Directory to write the LeRobot v2.1 dataset of the run into; `--record None` turns it off."""
    fps: int = 30
    task: str = "pick up the object, hold it, and set it down 5 cm away"


def look(args: Args) -> tuple[np.ndarray, CameraModel, list[Obj]]:
    cams = load_cameras()
    cmodel = cams[DET]
    if args.frames:
        frame = cv2.imread(args.frames)
        if frame is None:
            raise RuntimeError(f"could not read {args.frames}")
        objs = survey(frame, cmodel, args.min_area)
    else:  # never overwrite the file we were asked to plan from
        frames = grab_all(args)
        frame = frames[DET]
        CAPTURES.mkdir(exist_ok=True)
        for k, f in frames.items():
            cv2.imwrite(str(CAPTURES / ("survey.jpg" if k == DET else f"survey_{k}.jpg")), f)
        cams = still_cameras(frames, cams)
        frames = {k: f for k, f in frames.items() if k in cams}
        objs = survey_multi(frames, cams, args.min_area)
    objs = with_given(objs, args)
    if not objs:
        raise RuntimeError(f"nothing in the zone that is not in {BACKGROUND[DET].name} - either it is "
                           "empty, or the background was captured with the objects already standing on it")
    return frame, cmodel, objs


def with_given(objs: list[Obj], args: "Args") -> list[Obj]:
    """--given prepended as object 0; a detection within 6 cm of it is the same object and dropped."""
    if not args.given:
        return objs
    given = [tuple(float(v) for v in part.split(",")) for part in args.given.split(";") if part.strip()]
    out = [Obj("given" if i == 0 else f"given_obstacle{i}", (x, y), d, h, (230, 230, 230), measured=True)
           for i, (x, y, d, h) in enumerate(given)]
    if args.given_only:
        if objs:
            print(f"   (--given-only: ignoring {len(objs)} detection(s) - they are parts of the given objects)")
        return out
    for g in out:
        print(f"   using the given {'target' if g is out[0] else 'obstacle'}: {g.describe()}")
    return out + [o for o in objs if all(np.hypot(o.xy[0] - g.xy[0], o.xy[1] - g.xy[1]) > 0.06 for g in out)]


def det_index(args: "Args") -> int:
    return args.cam if DET == "cam0" else args.cam2


def stage_zone(args: Args) -> None:
    """Find the white sheet inside the blue tape in the detection camera's empty-zone background and store
    its corners in the robot frame (zone.json). Every stage then works inside it: only objects standing in
    it are detected, and objects are only ever set down in it."""
    bg = cv2.imread(str(BACKGROUND[DET]))
    if bg is None:
        raise RuntimeError(f"{BACKGROUND[DET]} missing: run --background with the zone empty first")
    cam = load_cameras()[DET]
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    white = cv2.morphologyEx(((hsv[..., 1] < 45) & (hsv[..., 2] > 150)).astype(np.uint8) * 255,
                             cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    blue = cv2.inRange(hsv, (95, 80, 60), (130, 255, 255))
    cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sheet = None
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:6]:  # the big white blob framed by blue tape
        fill = cv2.drawContours(np.zeros_like(white), [c], -1, 255, -1)
        ring = cv2.dilate(fill, np.ones((31, 31), np.uint8))
        ring[fill > 0] = 0
        if (blue[ring > 0] > 0).mean() > 0.5:
            sheet = c
            break
    if sheet is None:
        raise RuntimeError("no white area framed by blue tape in the background frame")
    quad = cv2.approxPolyDP(sheet, 0.02 * cv2.arcLength(sheet, True), True).reshape(-1, 2)
    poly = [[round(float(v), 4) for v in cam.hit_plane(float(u), float(w), 0.0)[:2]] for u, w in quad]
    sides = [float(np.linalg.norm(np.subtract(poly[i], poly[(i + 1) % len(poly)]))) for i in range(len(poly))]
    ZONE_FILE.write_text(json.dumps({"camera": DET, "polygon": poly, "pixels": quad.tolist(),
                                     "sides_m": [round(x, 4) for x in sides]}, indent=1))
    dbg = bg.copy()
    cv2.polylines(dbg, [quad.astype(np.int32)], True, (0, 0, 255), 3)
    cv2.imwrite(str(CAPTURES / "zone.jpg"), dbg)
    reach = [float(np.hypot(*c)) for c in poly]
    print(f"zone: {len(poly)} corners {poly}\n  sides {[round(x * 100, 1) for x in sides]} cm, "
          f"{min(reach):.2f}-{max(reach):.2f} m from the base -> {ZONE_FILE.name}, captures/zone.jpg")
    if min(reach) < REACH[0]:
        print(f"  note: part of the zone is nearer the base than {REACH[0]} m; objects there are refused")


def stage_scene(args: Args) -> None:
    """Recreate the 3D workspace map from every calibrated camera: their views, the carved objects and the
    zone, with the arm at REST (it is not read over CAN). -> scene.json + workspace_map.html"""
    from scene3d import camera_entry
    import live_map

    cams = load_cameras()
    raw = json.loads((HERE / "cameras.json").read_text())
    frames = grab_all(args)
    objs = survey_multi(frames, cams, args.min_area)
    for o in objs:
        print(f"   {o.describe()}")
    meshes = []  # live_map.build poses the arm itself; scene.json only needs the q
    scene = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frame": "robot base: x forward, y left, z up, table z=0 (m)",
        "cameras": [camera_entry(k, cams[k], frames[k], raw[k]) for k in sorted(cams)],
        "objects": [{"name": o.name, "axis": list(o.xy), "diameter": o.diameter, "height": o.height,
                     "color": list(o.color)} for o in objs],
        "robot": {"q": [0.0] * 6, "tcp": [0.0, 0.0, 0.0], "meshes": meshes},
        "calibration_points": [r["fk"] for d in ("sweep_zone2", "sweep_zone3")
                               for r in json.loads((CAPTURES / d / "sweep.json").read_text())],
    }
    zone = load_zone()
    if zone is not None:
        from fused import TopView, jpeg_b64
        tv = TopView(cams, zone)
        top = tv.annotate(tv.render(frames), zone, scene["objects"])
        cv2.imwrite(str(CAPTURES / "fused_top.jpg"), top)
        scene["top"] = {"b64": jpeg_b64(top), "meta": tv.meta()}
    (HERE / "scene.json").write_text(json.dumps(scene))
    print(f"wrote scene.json: {len(scene['cameras'])} cameras, {len(objs)} object(s)")
    live_map.build(HERE / "scene.json")


def stage_background(args: Args) -> None:
    """Capture the empty-table reference both this script and scene3d.py subtract. Everything that is on
    the table now becomes invisible to detection, so the table must be clear."""
    CAPTURES.mkdir(exist_ok=True)
    for key, idx in (("cam0", 0), ("cam1", 1), ("cam2", 2)):  # camN is OpenCV index N
        try:
            frame = grab(idx)
        except RuntimeError as e:
            print(f"{key}: {e} (skipped)")
            continue
        cv2.imwrite(str(BACKGROUND[key]), frame)
        print(f"wrote {BACKGROUND[key]}")
    print("the table must have been EMPTY for these: anything in them is invisible to --survey/--run")


def stage_survey(args: Args) -> None:
    _, _, objs = look(args)
    print(f"{len(objs)} object(s) on the table  (captures/survey.jpg)")
    for i, o in enumerate(objs):
        try:
            mode, reason = decide_grasp(o)
            verdict = f"{'VERTICAL' if mode == 'top' else 'HORIZONTAL'} pick - {reason}"
        except RuntimeError as e:
            verdict = f"NOT GRASPABLE - {e}"
        print(f"[{i}] {o.describe()}\n     {verdict}")


def plan_from_frame(frame: np.ndarray, cmodel: CameraModel, objs: list[Obj], args: Args) -> Plan:
    if not 0 <= args.object < len(objs):
        raise RuntimeError(f"--object {args.object} out of range: {len(objs)} object(s) found")
    print(f"{len(objs)} object(s) on the table")
    plan = make_plan(objs, args.object, args, AzPlanner())
    print_plan(plan)
    draw_plan(frame, cmodel, plan, CAPTURES / "plan.jpg")
    (HERE / "plan.json").write_text(json.dumps(plan.to_json(), indent=2))
    print("wrote plan.json and captures/plan.jpg")
    return plan


def stage_plan(args: Args) -> Plan:
    frame, cmodel, objs = look(args)
    plan = plan_from_frame(frame, cmodel, objs, args)
    print("(arm not touched)")
    return plan


def stage_run(args: Args) -> None:
    """One command, no prerequisites: look, plan, and - if the plan is safe - run it."""
    models = load_cameras()
    cmodel = models[DET]
    cams: dict[str, LiveCamera] = {}
    cam: LiveCamera | None = None
    if not args.sim:
        cams[DET] = cam = LiveCamera(det_index(args))
        for k in sorted(models):  # every calibrated camera: the survey carves the zone with all of them
            if k not in cams:
                cams[k] = LiveCamera(int(k[3:]))
        if args.record:  # record every camera, calibrated or not
            for i in range(3):
                if f"cam{i}" not in cams:
                    try:
                        cams[f"cam{i}"] = LiveCamera(i)
                    except RuntimeError as e:
                        print(f"cam{i}: {e} (not recorded)")
    planner = AzPlanner()
    live = LiveState(planner.model, None)
    live.set(status="planning", phase="connecting to the arm")
    try:
        # Connect before surveying. The parked arm is only cancelled by the background frame while it
        # stands exactly where it stood when that frame was taken; after an aborted run it is torqued off
        # and slumped somewhere else, and then it IS foreground - it read as a 2 cm object beside the base
        # and inflated the bottle's measured height from 12.5 to 17.0 cm, flipping the grasp decision from
        # vertical to horizontal. Its joint angles are known, so project it out of the frame instead.
        arm = Arm(args.sim, args.channel, args.hz)
    except Exception:
        for lc in cams.values():
            lc.close()
        raise
    try:
        frame = cam.latest() if cam is not None else cv2.imread(args.frames or str(CAPTURES / "survey.jpg"))
        CAPTURES.mkdir(exist_ok=True)
        cv2.imwrite(str(CAPTURES / "survey.jpg"), frame)
        if args.sim:
            objs = survey(frame, cmodel, args.min_area)
        else:  # every calibrated view, each with the parked arm projected out of it
            ws0, q0 = Workspace(planner, []), arm.q()
            frames = {k: cams[k].latest() for k in models if k in cams}
            models = still_cameras(frames, models)
            frames = {k: f for k, f in frames.items() if k in models}
            masks = {k: arm_mask(ws0, models[k], q0, f.shape[:2]) for k, f in frames.items()}
            for k, f in frames.items():
                if k != DET:
                    cv2.imwrite(str(CAPTURES / f"survey_{k}.jpg"), f)
            objs = survey_multi(frames, models, args.min_area, ignore=masks)
            FUSED_CAMS.update({k: cams[k] for k in models if k in cams})
            FUSED_MODELS.update(models)
            zone = load_zone()
            if zone is not None:  # the fused top view, re-rendered by LiveState for the map
                from fused import TopView, jpeg_b64
                tv = TopView(models, zone)
                live.top_fn = lambda: jpeg_b64(tv.annotate(tv.render({k: c.latest() for k, c in FUSED_CAMS.items()}),
                                                           zone, live.doc.get("objects", [])))
        objs = with_given(objs, args)
        if not objs:
            raise RuntimeError(f"nothing in the zone that is not in {BACKGROUND[DET].name}, and not the arm "
                               "itself - either the table is empty, or the background was captured with "
                               "the objects already standing on it")
        live.set(phase="planning")
        plan = plan_from_frame(frame, cmodel, objs, args)
        show_plan(live, planner, plan)
    except Exception as e:
        live.set(status="aborted", phase=f"no safe plan: {e}")
        arm.close()
        for lc in cams.values():
            lc.close()
        raise
    rec = None
    if args.record and cams:
        rec = EpisodeRecorder(Path(args.record), cams, arm.state7, lambda: arm.last_cmd, args.task, args.fps)
        rec.start()
        print(f"recording to {args.record} at {args.fps} fps")
    live.get_q7 = arm.state7
    live.set(status="running", phase="starting")
    live.start()
    ok = False
    try:
        ok = execute(plan, planner, arm, cam, cmodel, args, live)
    except Exception as e:
        print(f"!! {e}")
        live.set(phase=f"aborted: {e}")
        try:  # never leave the arm stretched out over the table
            safe_park(arm, planner, plan.yaw, plan.tilt)
        except Exception as e2:
            print(f"   could not park the arm: {e2}")
    finally:
        live.stop("done" if ok else "aborted")
        if rec:
            rec.stop()
        arm.close()
        for lc in cams.values():
            lc.close()
        if rec:  # save even a partial episode: it is the record of what the arm actually did
            try:
                out = rec.save()
                print(f"saved LeRobot dataset: {out} ({len(rec.states)} frames)")
                # the planned trajectory next to the recorded one: one line per episode
                with (Path(args.record) / "meta" / "episode_plans.jsonl").open("a") as fh:
                    fh.write(json.dumps({"episode_index": rec.ep, "completed": ok, "plan": plan.to_json(),
                                         "fingertip_trail": live.doc.get("trail", [])}) + "\n")
            except Exception as e:
                print(f"!! could not save the dataset: {e}")


def main(args: Args) -> None:
    global JAW_ACROSS
    if args.jaw_across:
        v = np.array([float(t) for t in args.jaw_across.split(",")])
        JAW_ACROSS = v / np.linalg.norm(v)
    if args.scene:
        stage_scene(args)
    elif args.zone:
        stage_zone(args)
    elif args.background:
        stage_background(args)
    elif args.survey:
        stage_survey(args)
    elif args.plan:
        stage_plan(args)
    elif args.run:
        stage_run(args)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main(tyro.cli(Args))
