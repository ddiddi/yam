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
FINGER_REACH = 0.081  # m: fingertip to the gripper housing's face, along the fingers (the arm model)


TOP_Z_MIN = 0.014  # m: a top grasp's fingertips never close lower (the side grasps run theirs at Z_TIP_MIN 1.2 cm)
_TILTS_OK: tuple = ()  # the SIDE_TILTS whose fingers cross the current target low enough (make_plan)
SIDE_TILTS: tuple = ()  # --side-tilt: side-grasp wrist tilts to try (rad from vertical), flattest first; () = SIDE_TILT
SIDE_CROSS_MAX = 0.6  # a side grasp's fingers must cross the object's axis within its lower 60%
SEAT_TIPS_ONLY = 9.9  # a SEAT_BACKOFF this large leaves no depth: the fingertips stop at the object's centre line
SEAT_BACKOFFS = (0.0, 0.015, 0.03, SEAT_TIPS_ONLY)  # the last: only for what nothing deeper can take (a flat dish)  # m: full depth first, then these much shallower if full depth has no safe path
SEAT_BACKOFF = 0.0  # the one the current plan was made (and must be executed) with


def seat_depth(obj: "Obj") -> float:
    """How far inside the jaw the object's near side sits, measured from the fingertips: as deep as the
    target clearance allows - its near side (tolerance + 5 mm) from the housing - so the fingers close on
    it along most of their length, from their base, not with the tips (less SEAT_BACKOFF when the full
    depth leaves no safe approach or retreat)."""
    tol = TALL_TOL if safe_height(obj) > TALL_H else TARGET_TOL
    return FINGER_REACH - tol - 0.005 - SEAT_BACKOFF
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


# ---- the weigh station (station.json, from calib_tools/station_calib.py + the measured footprint) -------
STATION_FILE = HERE / "station.json"
POUR: dict | None = None  # set by --pour-oz: {"oz", "base_z", "tip", "tip_at", "axis"}
STATION: dict | None = None  # set by --station: {"top_z", "footprint" [[x, y] x4], "place" [x, y], ...}
STATION_MARGIN = 0.012  # m: the station is a measured box, not a detection - a tighter margin than MARGIN
SURFACE_SLACK = 0.003  # m: how far below the station top a point may be over it (the carried object sits ON it)
STATION_DEPTH_BAND = 0.025  # m: a depth-only object centred this close to the station is its smeared edge


def load_station() -> dict:
    st = json.loads(STATION_FILE.read_text())
    for k in ("top_z", "footprint", "place"):
        if k not in st:
            raise RuntimeError(f"{STATION_FILE.name} has no '{k}' - measure the station first")
    return st


def poly_sd(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Signed distance of points to a convex polygon: positive outside, negative inside. Outside it is the
    largest edge-line distance, which never exceeds the true distance - clearances come out conservative."""
    P = np.asarray(poly, dtype=float)
    a, b = P[1] - P[0], P[2] - P[1]
    if a[0] * b[1] - a[1] * b[0] < 0:
        P = P[::-1]  # counter-clockwise: the outward normal of edge a->b is (dy, -dx)
    e = np.roll(P, -1, axis=0) - P
    n = np.c_[e[:, 1], -e[:, 0]] / np.linalg.norm(e, axis=1)[:, None]
    q = np.atleast_2d(np.asarray(xy, dtype=float))
    return ((q[:, None, :] - P[None, :, :]) * n[None, :, :]).sum(-1).max(axis=1)


def station_obj() -> "Obj | None":
    if STATION is None:
        return None
    fp = np.array(STATION["footprint"], dtype=float)
    return Obj("scale", tuple(fp.mean(axis=0)), 0.0, float(STATION["top_z"]), (40, 40, 40), measured=True,
               dims_from="station.json", poly=fp)


def surface_z(xy) -> float:
    """What an object at xy stands on: the station top over its footprint, the table elsewhere."""
    if STATION is not None and poly_sd(np.array(xy)[:2], np.array(STATION["footprint"]))[0] <= 0:
        return float(STATION["top_z"])
    return 0.0


def station_mask(cam: "CameraModel", shape: tuple[int, int], grow: int = 15) -> np.ndarray | None:
    """The station box (top and sides) in a camera image: detection ignores it. It is in the backgrounds, but
    an object set on it would otherwise read as one standing on the table, and depth would fit the table
    plane to its top."""
    if STATION is None:
        return None
    fp = np.array(STATION["footprint"], dtype=float)
    box = np.vstack([np.c_[fp, np.zeros(len(fp))], np.c_[fp, np.full(len(fp), STATION["top_z"])]])
    uv = cam.project(box)
    m = np.zeros(shape, np.uint8)
    if np.isfinite(uv).all():
        cv2.fillConvexPoly(m, cv2.convexHull(np.clip(uv, -1e5, 1e5).astype(np.int32)), 255)
    return cv2.dilate(m, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))


def box_hits(pts: np.ndarray, o: "Obj", margin: float) -> np.ndarray:
    """Points of pts (N x 3) inside a station box's no-go volume: below its top (less SURFACE_SLACK) and
    within `margin` of its footprint."""
    low = pts[:, 2] < o.height - SURFACE_SLACK
    out = np.zeros(len(pts), bool)
    if low.any():
        out[low] = poly_sd(pts[low, :2], o.poly) < margin
    return out


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
    # dimensional analysis: the footprint as an oriented rectangle (length >= width) and the long side's
    # direction (rad, robot frame). None = unmeasured, treated as round with `diameter`.
    length: float | None = None
    width: float | None = None
    angle: float | None = None
    dims_from: str = "single view (depth not measured: treated as round)"
    xy_uncertain: bool = False  # found by depth alone: its position along the line of sight is +-2 cm
    # a fixed box (the weigh station): its footprint polygon (N x 2, robot frame). `height` is its top, which
    # is a surface things are set down on - nothing may go below it over the footprint, anything may go above
    poly: np.ndarray | None = field(repr=False, default=None)
    # (z_from, diameter): above z_from the object is only this wide - a bottle's screw cap. With a neck the jaw
    # closes on the neck (the grasp height must be inside it) and the arm may come that close to the axis there
    neck: tuple | None = None

    @property
    def grip_width(self) -> float:
        """What the jaw must span: the neck when one is given, the footprint's short side when it is measured,
        else the diameter."""
        if self.neck is not None:
            return float(self.neck[1])
        return self.width if self.width is not None else self.diameter

    def radius_at(self, z) -> np.ndarray:
        """The no-touch radius at height(s) z above the object's base: the neck's above z_from."""
        z = np.asarray(z, dtype=float)
        if self.neck is None:
            return np.full(z.shape, self.radius)
        return np.where(z >= self.neck[0], self.neck[1] / 2, self.radius)

    @property
    def long_axis(self) -> np.ndarray | None:
        """Unit xy vector of the long side when the object is clearly elongated, else None (round)."""
        if self.length is None or self.width is None or self.angle is None:
            return None
        if self.length < ELONGATED * max(self.width, 1e-6):
            return None
        return np.array([np.cos(self.angle), np.sin(self.angle)])

    @property
    def radius(self) -> float:
        return self.diameter / 2

    def footprint(self, xy=None) -> tuple[np.ndarray, float]:
        """(centres, r): the footprint as circles - one for a round object; for an elongated one, circles of
        half its width strung along its axis (a 13.7 x 1.4 cm tube is not a 6.9 cm-radius disc)."""
        c = np.asarray(self.xy if xy is None else xy, dtype=float)
        ax = self.long_axis
        if ax is None:
            return c[None], self.radius
        half = max(self.length / 2 - self.width / 2, 0.0)
        n = max(2, int(np.ceil(2 * half / max(self.width, 0.01))) + 1)
        return np.array([c + t * ax for t in np.linspace(-half, half, n)]), self.width / 2

    @property
    def reach(self) -> float:
        return float(np.hypot(*self.xy))

    @property
    def flatness(self) -> float:
        """height / diameter: below 1 the object is flat and wide, well above 1 it is slim and upright."""
        return self.height / max(self.diameter, 1e-6)

    def describe(self) -> str:
        dims = ""
        if self.length is not None:
            dims = (f"\n            dims {self.length * 100:.1f} x {self.width * 100:.1f} x {self.height * 100:.1f} cm"
                    f" (L x W x H), long side at {np.degrees(self.angle):+.0f} deg - {self.dims_from}")
        else:
            dims = f"\n            dims {self.dims_from}"
        return (f"{self.name:<9} xy ({self.xy[0]:+.3f}, {self.xy[1]:+.3f})  d {self.diameter * 100:5.1f} cm"
                f"  h {self.height * 100:5.1f} cm  h/d {self.flatness:4.1f}  reach {self.reach:.2f} m"
                f"  rgb {tuple(self.color)}{dims}")


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


ELONGATED = 1.3  # length/width above this: the object has a short side the jaw must close across
# (a single view cannot measure depth: cam1 looks at the zone so shallowly that 1 cm of height error moves the
# far edge of a footprint by several cm - a 6.7 cm bottle read 14.6 cm long - so only the carved multi-view
# footprint, or a hand measurement, gives an object a short side)


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
                      "views": int(vmap[cells].max()), "cells": np.c_[gx[cells], gy[cells]]})
    return blobs


def carved_footprint(cells: np.ndarray) -> tuple[float, float, float] | None:
    """(length, width, long-side angle) of the smallest rectangle around a blob's solid columns."""
    if len(cells) < 4:
        return None
    (cx, cy), (a, b), ang = cv2.minAreaRect((cells * 1000.0).astype(np.float32))
    a, b = a / 1000.0 + VOXEL, b / 1000.0 + VOXEL  # column centres -> column edges
    th = np.radians(ang)
    if a < b:
        a, b, th = b, a, th + np.pi / 2
    return float(a), float(b), float((th + np.pi / 2) % np.pi - np.pi / 2)


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
            rect = carved_footprint(b["cells"])
            if rect is not None:
                o.length, o.width, o.angle = rect
                o.dims_from = f"carved from {b['views']} views"
                o.diameter = max(o.diameter, o.length)  # the clearance cylinder spans the long side
                print(f"   {o.name}: footprint {o.length * 100:.1f} x {o.width * 100:.1f} cm, long side at "
                      f"{np.degrees(o.angle):+.0f} deg ({'has a short side' if o.long_axis is not None else 'round'})")
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


DEPTH_WORLD: dict = {}  # the first scene analysis' metric height map (depth.py): H, lo, for the path check
DEPTH_MATCH = 0.06  # m: a depth object this close to a detected one is the same object


DEPTH_FRAMES = 5  # frames of the still scene the depth analysis must agree over


def fuse_depth(objs: list[Obj], frame: np.ndarray, cam: CameraModel, ignore: np.ndarray | None = None,
               key: str = DET, more: list | None = None) -> list[Obj]:
    """Depth Anything V2 (depth.py) on the detection camera, pinned to the calibrated table: every object's
    height and footprint from its own 3D points, and the objects background subtraction cannot see.

    What one view measures well: heights (+-1 cm), extents across the line of sight, the direction of a
    clearly long object. What it cannot: extent *along* the line of sight (the network's metric error, 2-3
    cm at this range, all lands there - a 4.8 cm deep box read 9 cm). So a footprint is only given a short
    side when it is clearly elongated (DEPTH_ELONGATED); otherwise the object stays round, sized by its
    larger side - the jaw then opens for it or the plan is refused, never guesses."""
    import depth

    try:
        with np.errstate(all="ignore"):
            r = depth.analyse_multi([frame] + list(more or []), cam, key, ignore=ignore)
    except Exception as e:
        print(f"   (depth analysis skipped: {e})")
        return objs
    DEPTH_WORLD.update(H=r["H"], lo=r["lo"])
    print(f"   depth (Depth Anything V2 Small, {r.get('frames', 1)} frame(s), {r['inference_s']} s): table fit rms "
          f"{r['fit']['table_rms_mm']:.1f} mm, {len(r['objects'])} stable object(s) -> captures/depth_{key}.jpg")
    for d in r.get("dropped", []):
        print(f"   (ignored a depth blob at {np.round(d['xy'], 3)}, {d['height'] * 100:.1f} cm: in {d['frames']}/"
              f"{r.get('frames', 1)} frames, wandering {d['spread_mm']} mm - not a real object)")
    used = set()
    for o in objs:
        ds = [float(np.hypot(*(np.array(d["xy"]) - np.array(o.xy)))) for d in r["objects"]]
        j = int(np.argmin(ds)) if ds else -1
        if j < 0 or ds[j] > DEPTH_MATCH or j in used:
            print(f"   {o.name}: no depth match - keeping the camera-survey measurement")
            continue
        used.add(j)
        _apply_depth(o, r["objects"][j], keep_xy=True)
    for j, d in enumerate(r["objects"]):
        if j in used:
            continue
        o = Obj(f"depth{j}", d["xy"], d["length"] + DEPTH_EDGE, d["height"], (200, 200, 200), xy_uncertain=True)
        _apply_depth(o, d, keep_xy=False)
        print(f"   NEW from depth only (white/clear on the sheet?): {o.describe()}")
        objs.append(o)
    return objs


DEPTH_EDGE = 0.01  # m: the depth-step filter shaves ~5 mm off each side of an object against the table
DEPTH_ELONGATED = 1.8  # length/width from one view's depth above which the long side's direction is trusted


def _apply_depth(o: Obj, d: dict, keep_xy: bool) -> None:
    L, W = d["length"] + DEPTH_EDGE, d["width"] + DEPTH_EDGE
    print(f"   {o.name}: depth says {L * 100:.1f} x {W * 100:.1f} x {d['height'] * 100:.1f} cm "
          f"(was h {o.height * 100:.1f} cm)")
    # depth measures tall things well but eats small ones (its edge filter strips a 3.5 cm wedge down to
    # 1.9 cm); a silhouette under-reads shiny caps. For an object both saw, the larger is the safer.
    o.height = float(d["height"]) if not keep_xy else max(o.height, float(d["height"]))
    o.measured = True
    if not keep_xy:
        o.xy = d["xy"]
    if L >= DEPTH_ELONGATED * W:
        o.length, o.width, o.angle = L, W, float(d["angle"])
        o.diameter = max(o.diameter, L)
        o.dims_from = "depth (one view): long side's direction and width measured"
    else:
        o.diameter = max(o.diameter, L)
        o.dims_from = (f"depth (one view): {L * 100:.1f} cm across, depth along the line of sight not "
                       "measurable - treated as round")


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


def grab_more(key: str, n: int) -> list[np.ndarray]:
    """n more frames of one camera, ~0.35 s apart (one open of the device)."""
    from scene3d import device_index  # YAM_CAM_MAP: after a replug camN is not always OpenCV index N

    cap = cv2.VideoCapture(device_index(int(key[3:])))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    out = []
    try:
        for _ in range(n):
            for _ in range(4):
                cap.grab()
            ok, f = cap.read()
            if ok:
                out.append(f)
            time.sleep(0.25)
    finally:
        cap.release()
    return out


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
        # each fingertip's slide (m): fully open unless told otherwise. A tilted wrist hangs one tip lower the
        # wider the jaw is, so legs travelled with the jaw only part-open are checked at that opening
        self.jaw_half = 0.0475
        self.is_tip = np.concatenate(self.tips)  # aligned with points(): True for fingertip vertices
        self.jaw_qpos = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                         for j in ("joint7", "joint8")
                         if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) >= 0]

    def geom_points(self, q6: np.ndarray) -> list[np.ndarray]:
        d = self.planner.data
        d.qpos[:] = 0
        d.qpos[:6] = q6
        for a in self.jaw_qpos:
            d.qpos[a] = self.jaw_half
        mujoco.mj_kinematics(self.planner.model, d)
        return [v @ d.geom_xmat[gi].reshape(3, 3).T + d.geom_xpos[gi] for gi, v in self.geoms]

    def points(self, q6: np.ndarray) -> np.ndarray:
        d = self.planner.data
        d.qpos[:] = 0
        d.qpos[:6] = q6
        for a in self.jaw_qpos:
            d.qpos[a] = self.jaw_half
        mujoco.mj_kinematics(self.planner.model, d)
        return np.concatenate([v @ d.geom_xmat[gi].reshape(3, 3).T + d.geom_xpos[gi] for gi, v in self.geoms])

    def clearance(self, pts: np.ndarray) -> tuple[float, str]:
        """Smallest gap between any point and any object's no-go cylinder (negative = inside it)."""
        worst, who = 1e9, ""
        for o in self.obstacles:
            if o.poly is not None:  # the station box: its top is a surface, only what goes below it counts
                low = pts[:, 2] < o.height - SURFACE_SLACK
                if low.any():
                    gap = float(poly_sd(pts[low, :2], o.poly).min()) - STATION_MARGIN
                    if gap < worst:
                        worst, who = gap, o.name
                continue
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
        xy, r, h, tol, obj, base = self.target
        P = pts[: len(self.is_tip)]
        sel = ~self.is_tip & (P[:, 2] < h + 0.01)
        if not sel.any():
            return 1e9
        rr = obj.radius_at(P[sel, 2] - base)  # a neck lets the arm closer up there
        return float((np.linalg.norm(P[sel, :2] - np.asarray(xy), axis=1) - rr).min()) - tol

    def scan(self, q_from: np.ndarray, q_to: np.ndarray, carry=None, n: int | None = None) -> tuple[float, str, float]:
        """Sweep the straight joint-space path the arm will actually glide through: worst clearance from
        the objects, what it was against, and the lowest point any mesh reaches (the table check).

        `Planner.path_is_safe` tests geom origins against a fixed 2 cm floor, which vetoes a legitimate
        low grasp on a flat object; the real fingertip surface is a mesh vertex, so it is measured here."""
        worst, who, zmin = 1e9, "", 1e9
        for a in np.linspace(0.0, 1.0, n or SCAN_N):
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
        """Where the target stands now (None while it hangs in the gripper). On the station its body starts at
        the station top, so its no-touch height is measured from there."""
        if obj is None:
            self.target = None
        else:
            h = safe_height(obj)
            tol = TALL_TOL if h > TALL_H else TARGET_TOL
            at = tuple(xy if xy is not None else obj.xy)
            self.target = (at, obj.radius, h + surface_z(at), tol, obj, surface_z(at))


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
    cs, fr = obj.footprint((0.0, 0.0))  # the orientation does not change while it is carried (same wrist yaw)
    ring = np.concatenate([np.stack([np.cos(th), np.sin(th)], axis=1) * (fr + 0.005) + c for c in cs])
    ring = ring + np.asarray(ahead, dtype=float)

    def f(tip: np.ndarray) -> np.ndarray:
        lo = max(tip[2] - z_grasp, 0.0)
        xy = ring + tip[:2]
        return np.array([[p[0], p[1], z] for z in np.linspace(lo, lo + obj.height, 4) for p in xy])

    return f


def grasp_half(obj: "Obj") -> float:
    """Half the object's extent along a side grasp's approach where the jaw closes: the neck's radius when the
    grasp is on a neck, half the length of a long object (entered along it), else the radius."""
    if obj.neck is not None:
        return float(obj.neck[1]) / 2
    return obj.length / 2 if obj.long_axis is not None else obj.radius


def side_deep(obj: "Obj", zg: np.ndarray) -> float:
    """Horizontal distance a side grasp's fingertips travel PAST the object's axis, so the axis crosses the
    fingers seat_depth - (half the object along the approach) from their tips: the object sits at the base
    of the fingers. zg is the (unit, 3D) approach direction; its horizontal part scales depth to xy."""
    half = grasp_half(obj)  # a long object is entered along its length; a neck is what the jaw closes on
    s = max(0.0, seat_depth(obj) - half)  # the axis' depth from the tips
    return float(s * np.linalg.norm(zg[:2]))


def grasp_ahead(planner: "AzPlanner", plan: "Plan") -> np.ndarray:
    """Where the object's axis sits relative to the fingertips once grasped (xy): for a side grasp it is
    behind them by side_deep (minus SLIDE_SHORT), towards the housing; nothing for a top grasp."""
    if plan.mode != "side":
        return np.zeros(2)
    keep = planner.az
    zg = planner.approach_dir(plan.pick, plan.yaw, plan.tilt, plan.az)
    planner.az = keep
    u = zg[:2] / (np.linalg.norm(zg[:2]) + 1e-9)
    return (SLIDE_SHORT - side_deep(plan.obj, zg)) * u


TIP_PROBE = 0.015  # --tip-probe: extra room assumed beyond each open fingertip when screening grasp poses
TABLE_SLACK = 0.004  # --table-slack: how far below the commanded fingertip height any part may go on a low leg


def table_floor(z_from: float, z_to: float) -> float:
    """Lowest the arm's meshes may reach on a leg: 1 cm normally, and just under the commanded
    fingertip height when that leg deliberately goes low (a flat object is grasped at ~1.2 cm)."""
    return min(0.01, min(z_from, z_to) - TABLE_SLACK)


def follow(arm: Arm, planner: AzPlanner, ws: "Workspace", qs: list, seconds: float, carry=None,
           floor_z: float | None = None) -> None:
    """Glide through a joint path `validate` produced, re-checking each segment against the obstacles, the
    target and the table as it goes (the scene may have been updated since). The first segment is scanned
    from the MEASURED joints, because that is where `Arm.glide` starts; after `settle` the commanded pose
    is deliberately off (it compensates sag) and the model puts it lower than the arm really is."""
    q = arm.q()
    ws.jaw_half = float(np.clip(arm.grip, 0.0, 1.0)) * JAW_STROKE / 2  # checked at the jaw's actual opening
    # a palm grasp leaves the arm touching what it just let go of: the first segment then starts "inside" it.
    # Leaving contact is allowed - never deeper than where it starts, and ending clearer than it began
    g_start = ws.scan(q, q, carry, n=1)[0]
    for k, q_next in enumerate(qs):  # the whole leg is re-checked first, then travelled as one smooth motion
        q_next = np.asarray(q_next, dtype=float)
        gap, who, zmin = ws.scan(q, q_next, carry)
        if floor_z is not None and zmin < floor_z:
            raise RuntimeError(f"a planned segment would reach {zmin * 100:.1f} cm: too low")
        if gap < 0:
            leaving = (k == 0 and g_start < 0 and gap >= g_start - 0.001
                       and ws.scan(q_next, q_next, carry, n=1)[0] > g_start)
            if not leaving:
                raise RuntimeError(f"a planned segment would enter {who}")
            print(f"   (leaving contact with {who}: {g_start * 100:+.1f} -> clearer)")
        q = q_next
    arm.glide_path([np.asarray(v, dtype=float) for v in qs], seconds)


def spline_samples(q0: np.ndarray, qs: list, n: int = 80) -> list[np.ndarray]:
    """The joint path Arm.glide_path will really travel through q0 -> qs (C2 spline over arc length), sampled."""
    Q = np.vstack([np.asarray(q0, dtype=float)[:6]] + [np.asarray(q, dtype=float)[:6] for q in qs])
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
    keep = np.r_[True, np.diff(d) > 1e-6]
    Q, d = Q[keep], d[keep]
    if len(Q) < 3:
        return [Q[0] + (Q[-1] - Q[0]) * t for t in np.linspace(0, 1, n)]
    from scipy.interpolate import CubicSpline
    spl = CubicSpline(d / d[-1], Q, axis=0, bc_type="natural")
    return [spl(t) for t in np.linspace(0, 1, n)]


def smooth_follow(arm: Arm, planner: AzPlanner, ws: "Workspace", qs: list, seconds: float, carry, floor_z: float) -> None:
    """Several validated legs as ONE continuous motion (no stop at the via points). The spline through them
    rounds the corners, so the path it really takes is swept against the objects and the table first."""
    ws.jaw_half = float(np.clip(arm.grip, 0.0, 1.0)) * JAW_STROKE / 2
    pts = spline_samples(arm.q(), qs)
    for a, b in zip(pts[:-1], pts[1:]):
        gap, who, zmin = ws.scan(a, b, carry, n=3)
        if zmin < floor_z:
            raise RuntimeError(f"the blended path would reach {zmin * 100:.1f} cm: too low")
        if gap < 0:
            raise RuntimeError(f"the blended path would enter {who}")
    arm.glide_path([np.asarray(q, dtype=float) for q in qs], seconds)


BLEND_KINDS = ("line", "joint", "pose")


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
    ws.jaw_half = float(np.clip(arm.grip, 0.0, 1.0)) * JAW_STROKE / 2
    path = []
    for i in range(1, n + 1):  # solved and checked end to end first, then one smooth motion
        p = p0 + (p1 - p0) * i / n
        q_next, _, _ = planner.ik(p[0], p[1], p[2], q, yaw, tilt)
        gap, who, zmin = ws.scan(q, q_next, carry)
        if zmin < floor:
            raise RuntimeError(f"segment towards {np.round(p, 3)} would reach {zmin * 100:.1f} cm: too low")
        if gap < 0:
            raise RuntimeError(f"segment towards {np.round(p, 3)} would enter {who}")
        path.append(q_next)
        q = q_next
    arm.glide_path(path, seconds)


def fmt_gap(g: float) -> str:
    """Clearances come back as a 1e9 sentinel when there is nothing to be clear of."""
    return "no other objects" if g > 1e6 else f"{g * 100:+.1f} cm"


def free_gap(xy, obstacles: list[Obj], extra: float = 0.0) -> tuple[float, str]:
    """Analytic clearance of a point (inflated by `extra`) from the no-go cylinders; a fast pre-filter."""
    worst, who = 1e9, ""
    for o in obstacles:
        if o.poly is not None:
            gap = float(poly_sd(np.asarray(xy)[:2], o.poly)[0]) - extra - STATION_MARGIN
        else:
            gap = float(np.linalg.norm(np.asarray(xy) - np.array(o.xy))) - o.radius - extra - MARGIN
        if gap < worst:
            worst, who = gap, o.name
    return worst, who


# ============================================================================== step 3: decision
def decide_grasp(obj: Obj) -> tuple[str, str]:
    """VERTICAL (top-down) or HORIZONTAL (side), from the object's flatness and height."""
    h, d, f = obj.height, obj.grip_width, obj.flatness
    if d > MAX_OBJ_WIDTH:
        side = "its narrowest side is" if obj.long_axis is not None else "it is"
        raise RuntimeError(f"{obj.name}: {side} {d * 100:.1f} cm, the jaw only spans "
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
        # full depth: the fingertips go down until the object's top is seat_depth inside the jaw (it was
        # 45% of the height, then 4.5 cm below the top - a pinch with the tips)
        return float(max(TOP_Z_MIN, obj.height - seat_depth(obj)))
    return float(np.clip(min(Z_SIDE_GRASP, 0.5 * obj.height), Z_TIP_MIN, 0.10))


def travel_height(obj: Obj, obstacles: list[Obj], z_grasp: float) -> float:
    """High enough that the carried object (which hangs z_grasp below the fingertips) clears everything,
    and that the fingertips hovering over the target on the way in clear its own top (they are exempt from
    the target clearance check, so a 25 cm wash bottle under a 20 cm hover was never flagged)."""
    # the target's own top gets 25% extra when it is tall: thin parts above its body (a wash bottle's spout,
    # a handle) are what depth and silhouettes miss, and the fingertips hover right over it
    top = safe_height(obj) * (1.25 if obj.height > TALL_H else 1.0)
    tallest = max([o.height for o in obstacles] + [top - z_grasp, 0.0])
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
    release: bool = False  # this waypoint lets go of the object (its grip may be only part-open)
    at: tuple | None = None  # where the object stands once this waypoint lets go of it (default: the place point)


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
    station: dict | None = None  # --station: the place point is the scale top; the object then goes back to `pick`
    weigh: float = 0.0  # --station: seconds the object sits on the scale, the gripper clear of it
    pour: dict | None = None  # --pour-oz: {"oz", "base_z", "tip"}; the place point is where the bottle axis pours from
    stage: str = "cycle"  # --station-stage: "cycle" (on, weigh, back), "on" (set it on the scale, weigh, leave it), "off"
    # (it already stands on the scale: take it off and set it down at `pick`)

    def to_json(self) -> dict:
        return {
            "object": {"name": self.obj.name, "xy": list(self.obj.xy), "diameter": self.obj.diameter,
                       "height": self.obj.height, "flatness": self.obj.flatness},
            "obstacles": [{"name": o.name, "xy": list(o.xy), "diameter": o.diameter, "height": o.height,
                           **({"footprint": np.asarray(o.poly).tolist()} if o.poly is not None else {})}
                          for o in self.obstacles],
            "station": self.station,
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
                    open_grip: float, hold: float, pause: float = 0.0, park: str = "rest",
                    station: dict | None = None, weigh: float = 0.0, pour: dict | None = None,
                    stage: str = "cycle") -> list[WP]:
    """With `station`: the place point is on the scale top (every place height rises by its top_z); after the
    release the gripper backs off and waits `weigh` s, takes the object again exactly as it let go of it, and
    sets it down back where it was picked (`at` on each release says where the object then stands)."""
    x, y = pick
    px, py = place
    dz = float(station["top_z"]) if station else 0.0  # the place surface's height
    zp = z_grasp + dz  # fingertip height at the place point
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
        # full depth: the fingertips then go on PAST the axis (side_deep) so the object ends up against the
        # base of the fingers, not pinched by the tips; the pre-grasp backs off far enough that the tips
        # still start clear of the object's near side
        u = np.r_[zg[:2] / (np.linalg.norm(zg[:2]) + 1e-9), 0.0]
        shift = (side_deep(obj, zg) - SLIDE_SHORT) * u
        x, y = x + shift[0], y + shift[1]
        px, py = px + shift[0], py + shift[1]
        half = grasp_half(obj)
        back = max(SIDE_APPROACH, seat_depth(obj) + half + 0.04)
        pre = tuple(np.array([x, y, z_grasp]) - back * zg)
        pre_place = tuple(np.array([px, py, zp]) - back * zg)
        on = "onto the scale" if station else "the place point"
        if pour:  # hold it over the bowl (the place point is where its axis goes), squeeze, bring it back
            zp = z_grasp + float(pour["base_z"])  # the bottle's base hangs this high while it pours
            wps += [
                WP("pre-grasp beside the object", pre, 3.0, settle=True),
                WP("slide the fingers around it", (x, y, z_grasp), 3.0, settle=True),
                WP("close on the object", (x, y, z_grasp), 0.0, kind="close"),
                WP("lift", (x, y, z_travel), 3.0),
                WP(f"hold {hold:.0f} s", (x, y, z_travel), hold, kind="hold"),
            ] + look2 + [
                WP("carry over the bowl", (px, py, z_travel), 3.5),
                WP("lower to the pour height", (px, py, zp), 3.0, settle=True),
                WP(f"squeeze-pour {pour['oz']:.1f} oz", (px, py, zp), 0.0, kind="squeeze"),
                WP("lift from the bowl", (px, py, z_travel), 3.0),
                WP("carry back to where it was", (x, y, z_travel), 3.5),
                WP("lower back down", (x, y, z_grasp), 3.0, settle=True),
                WP("release it back", (x, y, z_grasp), 1.5, grip=GRIP_OPEN, at=tuple(pick)),
                WP("back the fingers out again", pre, 2.5),
                WP("clear of the table", (x, y, z_travel), 2.5),
            ]
            wps.append(WP("fold back to ready", ready, 3.5, kind="pose", q=np.array(READY)))
            if park == "rest":
                wps.append(WP("return to rest", (0.0, 0.0, 0.0), 3.0, kind="rest"))
            return wps
        wps += [
            WP("pre-grasp beside the object", pre, 3.0, settle=True),
            WP("slide the fingers around it", (x, y, z_grasp), 3.0, settle=True),
            WP("close on the object", (x, y, z_grasp), 0.0, kind="close"),
            WP("lift", (x, y, z_travel), 3.0),
            WP(f"hold {hold:.0f} s", (x, y, z_travel), hold, kind="hold"),
        ] + look2 + [
            WP("carry onto the scale" if station else "carry to the place point", (px, py, z_travel), 3.5),
            WP("lower onto the scale" if station else "lower", (px, py, zp), 3.0, settle=True),
            WP("release", (px, py, zp), 1.5, grip=GRIP_OPEN, at=tuple(place)),
            WP("back the fingers out", pre_place, 2.5),
            WP("clear of the scale" if station else "clear of the table", (px, py, z_travel), 2.5),
        ]
        if station:
            wps += [
                WP(f"weigh {weigh:.0f} s", (px, py, z_travel), weigh, kind="hold"),
                WP("back beside it on the scale", pre_place, 3.0, settle=True),
                WP("slide the fingers around it again", (px, py, zp), 3.0, settle=True),
                WP("close on it on the scale", (px, py, zp), 0.0, kind="close"),
                WP("lift off the scale", (px, py, z_travel), 3.0),
                WP("carry back to where it was", (x, y, z_travel), 3.5),
                WP("lower back down", (x, y, z_grasp), 3.0, settle=True),
                WP("release it back", (x, y, z_grasp), 1.5, grip=GRIP_OPEN, at=tuple(pick)),
                WP("back the fingers out again", pre, 2.5),
                WP("clear of the table", (x, y, z_travel), 2.5),
            ]
    else:
        wps += [
            WP("above the grasp height", (x, y, z_grasp + 0.03), 3.0, settle=True),
            WP("descend onto the object", (x, y, z_grasp), 1.5, settle=True),
            WP("close on the object", (x, y, z_grasp), 0.0, kind="close"),
            WP("lift", (x, y, z_travel), 3.0),
            WP(f"hold {hold:.0f} s", (x, y, z_travel), hold, kind="hold"),
        ] + look2 + [
            WP("carry onto the scale" if station else "carry to the place point", (px, py, z_travel), 3.5),
            WP("lower onto the scale" if station else "lower", (px, py, zp), 3.0, settle=True),
            # let go only as wide as the pick opening: a tilted wrist opened fully this low puts a fingertip
            # into the table. It opens fully once it is up.
            WP("release", (px, py, zp), 1.5, grip=open_grip, release=True, at=tuple(place)),
            WP("clear of the scale" if station else "clear of the table", (px, py, z_travel), 2.5, grip=GRIP_OPEN),
        ]
        if station:
            wps += [
                WP(f"weigh {weigh:.0f} s", (px, py, z_travel), weigh, kind="hold"),
                # back down around it at the pick opening again (fully open, a tilted wrist hangs a tip low)
                WP("above it on the scale", (px, py, zp + 0.03), 3.0, settle=True, grip=open_grip),
                WP("descend onto it on the scale", (px, py, zp), 1.5, settle=True),
                WP("close on it on the scale", (px, py, zp), 0.0, kind="close"),
                WP("lift off the scale", (px, py, z_travel), 3.0),
                WP("carry back to where it was", (x, y, z_travel), 3.5),
                WP("lower back down", (x, y, z_grasp), 3.0, settle=True),
                WP("release it back", (x, y, z_grasp), 1.5, grip=open_grip, release=True, at=tuple(pick)),
                WP("clear of the table", (x, y, z_travel), 2.5, grip=GRIP_OPEN),
            ]
    if station and stage != "cycle":
        iw = next(i for i, w in enumerate(wps) if w.label.startswith("weigh"))
        if stage == "on":  # leave it on the scale once weighed
            wps = wps[:iw + 1]
        else:  # "off": it already stands on the scale - go straight to it, then the second half of the cycle
            wps = [wps[0], WP("above it on the scale", (px, py, z_travel), 4.0, kind="joint")] + wps[iw + 1:]
    wps.append(WP("fold back to ready", ready, 3.5, kind="pose", q=np.array(READY)))
    if park == "rest":
        wps.append(WP("return to rest", (0.0, 0.0, 0.0), 3.0, kind="rest"))
    return wps


def starts_placed(plan: "Plan") -> bool:
    """--station-stage off: the target starts on the scale (at `place`), not at `pick`."""
    return plan.station is not None and plan.stage == "off"


# --fast: the light planner. The same collision sweep, coarser: IK every VALIDATE_STEP along straight legs and
# SCAN_N poses per joint segment; no 1 cm robustness re-plans, no fused-view check, no depth, no rescans.
FAST = False
VALIDATE_STEP = 0.02  # m between IK solves along a straight leg (--fast: 0.05)
SCAN_N = 12  # poses swept per joint-space segment (--fast: 4)


def validate(planner: AzPlanner, ws: Workspace, plan: Plan, q_start: np.ndarray, step: float | None = None,
             wps: list[WP] | None = None, holding: bool = False,
             placed: bool = False) -> tuple[float, str, list[tuple[str, float]], float]:
    """Walk the waypoints exactly as `execute` will - IK on every 2 cm sub-step, then sweep each joint
    segment against the no-go cylinders. Returns the worst clearance, what it was against, the clearance
    of every leg, and the lowest point any part of the arm reaches."""
    planner.az = plan.az
    step = VALIDATE_STEP if step is None else step
    carry = carried_points(plan.obj, plan.z_grasp, grasp_ahead(planner, plan))
    q = np.array(q_start[:6])
    worst, who, legs, low = 1e9, "", [], 1e9
    keep = ws.target
    placed = placed or starts_placed(plan)
    ws.aim(None if holding else plan.obj, None if holding else (plan.place if placed else plan.pick))
    try:
        return _walk(planner, ws, plan, q, step, wps, holding, carry, worst, who, legs, low)
    finally:
        ws.target = keep


def _walk(planner, ws, plan, q, step, wps, holding, carry, worst, who, legs, low):
    part_open = plan.open_grip * JAW_STROKE / 2  # the jaw's slide per tip until it is opened fully
    full_open = False
    for wp in (plan.waypoints if wps is None else wps):
        ws.jaw_half = 0.0475 if (full_open or wp.kind in ("rest", "pose")) else max(part_open, 0.0)
        if wp.kind == "close":
            holding = True
            ws.aim(None)
            continue
        if wp.kind in ("hold", "rescan", "squeeze"):
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
            full_open = True
        if wp.release or (wp.grip is not None and wp.grip >= GRIP_OPEN - 1e-6):
            if holding:
                ws.aim(plan.obj, wp.at or plan.place)  # released: it stands where this leg set it down
            holding = False
        legs.append((wp.label, gap))
        low = min(low, zmin)
        if gap < worst:
            worst, who = gap, w
    return worst, who, legs, low


JAW_ACROSS: np.ndarray | None = None  # unit xy direction the jaw must close across (--jaw-across)
JAW_ALIGN_DEG = 12.0  # how far off square to the long side the jaw may close


def pose_options(mode: str) -> list[tuple[float, float, float | None]]:
    """(yaw, tilt, azimuth offset) candidates. For a top grasp the approach azimuth is redundant with the
    wrist yaw (the wrist points down), so the yaw is spun instead - half a turn, the jaw being symmetric -
    over the small outward tilts `pick_bottle` also needs, because a strictly vertical wrist runs into the
    wrist-pitch limit close to the base. A side grasp instead needs the direction the fingers come in
    from, which is searched around the radial one."""
    if mode == "top":
        # the jaw line repeats every half turn, but the wrist joint does not: a yaw past its limit can be
        # reached half a turn round (a tube at 156 deg only fitted at yaw 135, not 315), so try the full turn
        return [(float(np.pi + a), float(t), None)
                for t in (0.0, 0.25, 0.45) for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
    tilts = _TILTS_OK or SIDE_TILTS or (SIDE_TILT,)
    return [(np.pi, float(t), float(d)) for t in tilts for d in (0.0, 0.35, -0.35, 0.7, -0.7, 1.05, -1.05, 1.4, -1.4)]


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
        across = JAW_ACROSS if JAW_ACROSS is not None else obj.long_axis
        if across is not None and abs(float(jd @ across)) > np.sin(np.radians(JAW_ALIGN_DEG)):
            continue  # the jaw would close along the long side: too wide, or fingertips landing on a handle
        half = obj.grip_width / 2 if (across is not None or obj.neck is not None) else obj.radius
        probes = [np.array(pick) + s * (half + JAW_GAP + TIP_PROBE) * jd for s in (1, -1)]  # the two tips
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
            tip = np.array([pick[0], pick[1], z_grasp])
            if mode == "side":  # where the fingertips really go: past the axis, the object at the finger base
                zg = planner.approach_dir(pick, yaw, tilt, az)
                tip[:2] += side_deep(obj, zg) * zg[:2] / (np.linalg.norm(zg[:2]) + 1e-9)
            planner.ik(tip[0], tip[1], tip[2], REST, yaw, tilt)
            if mode == "side":
                half_a = grasp_half(obj)
                pre = tip - max(SIDE_APPROACH, seat_depth(obj) + half_a + 0.04) * zg
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
        cs, fr = obj.footprint(p)
        if zone is not None and min(zone_margin(c, zone) for c in cs) < fr + 0.005:
            continue  # the whole footprint must land inside the zone
        gap = min(free_gap(c, ws.obstacles, fr)[0] for c in cs)
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


def as_circles(o: Obj) -> list[Obj]:
    """An elongated obstacle as a chain of short-side-wide cylinders along its length (the planner's
    clearance model is cylinders); a round one - or the station box - as itself."""
    ax = o.long_axis
    if ax is None or o.poly is not None:
        return [o]
    n = int(np.ceil((o.length - o.width) / (o.width / 2))) + 1
    return [Obj(f"{o.name}[{k}]", (o.xy[0] + t * ax[0], o.xy[1] + t * ax[1]), o.width, o.height, o.color)
            for k, t in enumerate(np.linspace(-(o.length - o.width) / 2, (o.length - o.width) / 2, n))]


def make_plan(objs: list[Obj], index: int, args: "Args", planner: AzPlanner) -> Plan:
    """The deepest grasp that has a safe path: full depth (the object against the finger base), then
    SEAT_BACKOFFS shallower. SEAT_BACKOFF stays at the chosen value for the execution."""
    global SEAT_BACKOFF
    errors = []
    for b in SEAT_BACKOFFS:
        SEAT_BACKOFF = b
        try:
            plan = _make_plan(objs, index, args, planner)
        except RuntimeError as e:
            errors.append(f"seated {seat_depth(objs[index]) * 100:.1f} cm deep: {e}")
            continue
        o = plan.obj
        if b >= SEAT_TIPS_ONLY:
            print("  seat: SHALLOWEST - the fingertips stop at the object's centre line; no deeper grasp was possible "
                  "(for a flat object, a deeper side grasp meets it above its rim)")
            return plan
        if plan.mode == "top":  # what is between the fingers: from the tips up to the object's top
            d = min(seat_depth(o), max(0.0, o.height - plan.z_grasp))
            where = f"its top {(FINGER_REACH - d) * 100:.1f} cm from the finger base"
            if o.height - plan.z_grasp < seat_depth(o):
                where += f"; the fingertips stop {plan.z_grasp * 100:.1f} cm above the table, so it cannot sit deeper"
        else:
            d = seat_depth(o)
            where = f"its back {(FINGER_REACH - d) * 100:.1f} cm from the finger base"
        print(f"  seat: the fingers wrap {d * 100:.1f} cm of their {FINGER_REACH * 100:.1f} cm length around the "
              f"object ({where})" + (f" - {b * 100:.1f} cm short of full depth: the full-depth path was not safe"
                                     if b else " - full depth"))
        return plan
    SEAT_BACKOFF = 0.0
    raise RuntimeError(" || ".join(errors))


def _make_plan(objs: list[Obj], index: int, args: "Args", planner: AzPlanner) -> Plan:
    obj = objs[index]
    obstacles = [c for i, o in enumerate(objs) if i != index for c in as_circles(o)]
    st_obj = station_obj()
    if st_obj is not None:
        obstacles.append(st_obj)
        # the object must stand on the scale top, not hang over its edge
        cs, fr = obj.footprint(STATION["place"])
        over = float(poly_sd(cs, st_obj.poly).max()) + fr
        if over > -0.01:
            raise RuntimeError(f"{obj.name} would not fit on the scale top at {STATION['place']}: its footprint comes "
                               f"{(over + 0.01) * 100:.1f} cm too close to the edge")
    if not (REACH[0] <= obj.reach <= REACH[1]):
        raise RuntimeError(f"{obj.name} is {obj.reach:.2f} m from the base: outside the safe reach {REACH}")
    mode, reason = decide_grasp(obj)
    if args.grasp != "auto":
        mode, reason = args.grasp, f"forced by --grasp {args.grasp}"
    ws = Workspace(planner, obstacles, args.margin)
    modes = [mode] if args.grasp != "auto" else [mode] + [m for m in ("top", "side") if m != mode]
    if args.grasp == "side" and not args.side_only:
        # when no side grasp can seat the object deep in the jaw (a long object entered along its length, or a
        # low one the 60-degree fingers would cross above its top), take it from above - still full depth,
        # still across the short side - rather than pinch it or not pick it at all
        modes.append("top")
    tried: list[str] = []
    fallback = None  # a plan that passes as detected but not with the pick shifted by a centimetre
    for m in modes:
        z0 = args.z_grasp if args.z_grasp is not None else grasp_height(m, obj)
        # a side grasp at the very bottom may leave the housing too close to a tall body: try a little higher
        zs = [z0] if (args.z_grasp is not None or m != "side") else [z0, z0 + 0.015, z0 + 0.03]
        for z_grasp in zs:
            if m == "side":
                # the fingers slope down to their tips (SIDE_TILT from vertical): at full depth they cross the
                # object's axis this much higher than the tips - it must still be the object's lower part
                half = grasp_half(obj)
                global _TILTS_OK
                tilts = SIDE_TILTS or (SIDE_TILT,)
                rise = max(0.0, seat_depth(obj) - half)
                _TILTS_OK = tuple(t for t in tilts if z_grasp + rise * np.cos(t) <= args.side_cross_max * obj.height)
                cross = z_grasp + rise * np.cos(max(tilts))  # the flattest wrist crosses lowest
                if not _TILTS_OK:
                    tried.append(f"side at {z_grasp * 100:.1f} cm: the fingers would cross the object at "
                                 f"{cross * 100:.1f} cm, above {args.side_cross_max:.0%} of its {obj.height * 100:.1f} cm")
                    continue
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
                    if POUR is not None:
                        place = tuple(float(v) for v in POUR["axis"])
                    elif STATION is not None:
                        place = tuple(float(v) for v in STATION["place"])
                    elif args.shuttle and place_dir is None:  # repeated runs go back and forth between two spots
                        zone = load_zone()
                        if zone is not None:  # towards the zone's middle; from the middle, back out the way it came
                            to_mid = zone.mean(axis=0) - np.array(obj.xy)
                            ang = float(np.degrees(np.arctan2(to_mid[1], to_mid[0])))
                            place_dir = ang if np.linalg.norm(to_mid) > args.place_dist / 2 else ang + 180.0
                        else:
                            place_dir = 0.0 if obj.reach < SHUTTLE_R else 180.0
                    if STATION is None and POUR is None:
                        try:
                            place = choose_place(planner, ws, obj, obj.xy, yaw, tilt, az, z_grasp, z_travel,
                                                 args.place_dist, place_dir)
                        except RuntimeError:
                            if place_dir is None or args.place_dir is not None:
                                raise
                            # the shuttle's preferred direction is blocked: any free direction will do
                            place = choose_place(planner, ws, obj, obj.xy, yaw, tilt, az, z_grasp, z_travel,
                                                 args.place_dist, None)
                    gap_side = TALL_JAW_GAP if (safe_height(obj) > TALL_H or obj.xy_uncertain) else JAW_GAP
                    open_grip = float(min(GRIP_OPEN, (obj.grip_width + 2 * gap_side) / JAW_STROKE))
                    why = reason if m == mode else f"{reason} -- BUT that failed ({tried[-1]}), fell back to '{m}'"
                    plan = Plan(obj, obstacles, m, why, yaw, tilt, az, obj.xy, place, z_grasp, z_travel, open_grip,
                                station=STATION if POUR is None else None,
                                weigh=args.weigh if STATION is not None else 0.0, pour=POUR,
                                stage=args.station_stage if (STATION is not None and POUR is None) else "cycle")
                    pause = args.pause if args.rescan else 0.0
                    for park in ("rest", "ready"):
                        plan.waypoints = build_waypoints(planner, obj, m, yaw, tilt, az, obj.xy, place, z_grasp,
                                                         z_travel, open_grip, args.hold, pause, park,
                                                         plan.station, plan.weigh, plan.pour, plan.stage)
                        gap, who, legs, low = validate(planner, ws, plan, REST)
                        # something parked against the base only blocks the way home: end upright at READY instead
                        if gap >= 0 or park == "ready" or [l for l, g in legs if g < 0] != ["return to rest"]:
                            break
                        print("  the way back to the folded rest pose is blocked; the run will park at READY instead")
                    if gap < 0:
                        leg = min(legs, key=lambda t: t[1])[0]
                        raise RuntimeError(f"'{leg}' passes {abs(gap) * 100:.1f} cm inside {who}'s no-go zone"
                                           + (f" (the arm already stands that close to {who} at rest; move it)"
                                              if leg == "unfold to ready" else ""))
                    if POUR is not None and args.pour_tilt_deg > 0:
                        # the tilt toward the bowl is planned at run time from the arm's pose; check now that it exists
                        lw = next(w for w in plan.waypoints if w.label == "lower to the pour height")
                        base = np.array([place[0], place[1], float(POUR["base_z"])])
                        tip = base + np.array(POUR["tip"])
                        lip = np.array([POUR["tip_at"][0], POUR["tip_at"][1], args.pour_lip])
                        bowl = tuple(float(v) for v in args.pour_bowl.split(",")) if args.pour_bowl else None
                        q_lv = lw.qs[-1] if lw.qs else planner.ik(*lw.xyz, np.array(READY), yaw, tilt)[0]
                        if bottle_tilt_path(planner, ws, np.asarray(q_lv)[:6], base, tip, lip,
                                            np.radians(args.pour_tilt_deg), bowl) is None:
                            raise RuntimeError(f"no safe {args.pour_tilt_deg:.0f} deg tilt toward the bowl")
                        print(f"  tilt toward the bowl: {args.pour_tilt_deg:.0f} deg, spout tip down to "
                              f"{args.pour_lip * 100:.0f} cm - reachable and clear")
                    plan.clearance, plan.tight_at, plan.legs, plan.low_point = gap, who, legs, low
                    plan.margin = args.margin
                    if args.servo:
                        check_servo_point(planner, ws, plan)
                    weak = "" if FAST else fragile(plan, planner, ws, args)
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
        place = plan.place if (plan.station is not None or plan.pour is not None) else tuple(np.array(plan.place) + s)
        alt = rebuild(plan, planner, tuple(np.array(plan.pick) + s), place, args.hold, pause)
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
             plan.low_point, plan.margin, plan.servo_ok, plan.servo_note, plan.servo_dist, [], plan.servo_dir,
             plan.station, plan.weigh, plan.pour, plan.stage)
    p.waypoints = build_waypoints(planner, plan.obj, plan.mode, plan.yaw, plan.tilt, plan.az, pick, place,
                                  plan.z_grasp, plan.z_travel, plan.open_grip, hold, pause,
                                  station=plan.station, weigh=plan.weigh, pour=plan.pour, stage=plan.stage)
    return p


def replan(plan: Plan, planner: AzPlanner, pick: tuple[float, float], hold: float, pause: float) -> Plan:
    """Shift the plan onto a corrected pick point (after the visual check); the place point rides along
    on the same offset, which was already validated against the obstacles."""
    if plan.station is not None or plan.pour is not None:  # the scale / the bowl does not move with the pick
        return rebuild(plan, planner, pick, plan.place, hold, pause)
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
    place = plan.place if (plan.station is not None or plan.pour is not None) else tuple(float(v) for v in np.array(cmd) + off)
    base = Plan(plan.obj, plan.obstacles, plan.mode, plan.reason, yaw, tilt, az, cmd,
                place, plan.z_grasp, plan.z_travel, plan.open_grip,
                [], plan.clearance, plan.tight_at, plan.legs, plan.low_point, plan.margin, plan.servo_ok,
                plan.servo_note, plan.servo_dist, [], plan.servo_dir, plan.station, plan.weigh, plan.pour)
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
          f"{plan.open_grip * JAW_STROKE * 100:.1f} cm for a {plan.obj.grip_width * 100:.1f} cm "
          f"{'short side' if plan.obj.long_axis is not None else 'object'}")
    print("\nstep 2  trajectory")
    print(f"  travel height {plan.z_travel * 100:.0f} cm, wrist yaw {np.degrees(plan.yaw):.0f} deg, tilt "
          f"{np.degrees(plan.tilt):.0f} deg, approach azimuth "
          f"{'radial' if plan.az is None else f'{np.degrees(plan.az):.0f} deg'}")
    moves = [w for w in plan.waypoints if w.kind not in ("hold", "close", "rescan", "squeeze")]
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
    out, notes = list(fresh) + [k for k in known if k.poly is not None], []  # the station box is fixed
    known = [k for k in known if k.poly is None]
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
        sm = station_mask(models[k], f.shape[:2])
        if sm is not None:  # the scale - and whatever stands on it - is not an obstacle on the table
            m |= sm
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
    if plan.station is not None:
        print("   (the place point is the scale: there is no other one to try)")
        return plan, False
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
OPEN_CAMS: dict = {}  # every camera the run has open (recording), fused or not
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
                 "role": "station" if o.poly is not None else "obstacle",
                 **({"footprint": np.asarray(o.poly).round(4).tolist()} if o.poly is not None else {})}
                for o in plan.obstacles])


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
    holding, placed = False, starts_placed(plan)
    stands = tuple(plan.place if placed else plan.pick)  # where the target stands while it is not in the gripper
    swept, bad = [], []
    why = {"obstacles": 0, "depth height map": 0, "target": 0, "carried object": 0}
    obst = [(np.array(o.xy), o.radius + ws.margin, o.height + 0.01) for o in plan.obstacles if o.poly is None]
    boxes = [o for o in plan.obstacles if o.poly is not None]
    # the depth height map: every cell that rises above the sheet, detected as an object or not. Cells that
    # belong to the target (inside its circle + tolerance) are left to the target test below.
    hm = None
    if DEPTH_WORLD.get("H") is not None:
        import depth as _d
        H, lo = np.nan_to_num(DEPTH_WORLD["H"], nan=0.0), DEPTH_WORLD["lo"]
        ij = np.argwhere(H > _d.MIN_RISE)
        cxy = lo + (ij + 0.5) * _d.CELL
        # the target's cells: around where it IS (plan.obj.xy, the world) as well as around the pick point, which
        # the visual check's correction moves into the arm's own frame (1.8 cm on the run that found this)
        r_t = plan.obj.radius + TALL_TOL + 0.01
        mine = ((np.linalg.norm(cxy - np.array(plan.pick), axis=1) < r_t) |
                (np.linalg.norm(cxy - np.array(plan.obj.xy), axis=1) < r_t))
        Hn = H.copy()
        for i, j in ij[mine]:
            Hn[i, j] = 0.0
        # ... and every raised patch connected to the target's cells: its depth trail and front face are the
        # target too (a 19 cm bottle's patch reached past the circle and flagged 296 arm points)
        n_c, lab = cv2.connectedComponents((H > _d.MIN_RISE).astype(np.uint8), connectivity=8)
        for k in set(int(lab[i, j]) for i, j in ij[mine]) - {0}:
            Hn[lab == k] = 0.0
        r = int(np.ceil(ws.margin / _d.CELL))  # a cell's top counts within the margin around it
        hm = (lo, _d.CELL, cv2.dilate(Hn.astype(np.float32), np.ones((2 * r + 1, 2 * r + 1), np.uint8)))
    th = safe_height(plan.obj)
    tol = TALL_TOL if th > TALL_H else TARGET_TOL
    full_open = False  # the jaw is swept at the opening each leg really has (as in validate)
    keep_jaw = ws.jaw_half
    for wp in walk:
        ws.jaw_half = 0.0475 if (full_open or wp.kind in ("rest", "pose")) else plan.open_grip * JAW_STROKE / 2
        if wp.grip is not None and wp.grip >= GRIP_OPEN - 1e-6:
            full_open = True  # opens fully once this leg arrives: the legs after it sweep it open
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
                for o in boxes:
                    b |= box_hits(pts, o, STATION_MARGIN)
                why["obstacles"] += int(b.sum())
                if hm is not None:
                    bh = ~tip & _below_heightmap(pts, hm)
                    why["depth height map"] += int((bh & ~b).sum())
                    b |= bh
                if not holding:  # the target standing where it was last set down (on the table or the scale)
                    c = np.array(stands)
                    base = surface_z(stands)
                    bt = ~tip & (pts[:, 2] < th + base + 0.01) & \
                        (np.linalg.norm(pts[:, :2] - c, axis=1) < plan.obj.radius_at(pts[:, 2] - base) + tol)
                    why["target"] += int((bt & ~b).sum())
                    b |= bt
                if holding:
                    cp = carry(planner.fk_pos(qq))
                    cb = np.zeros(len(cp), bool)
                    for c, r, h in obst:
                        cb |= (cp[:, 2] < h) & (np.linalg.norm(cp[:, :2] - c, axis=1) < r)
                    for o in boxes:  # the carried object's base sits ON the scale top when it is set down
                        cb |= box_hits(cp, o, STATION_MARGIN)
                    why["carried object"] += int(cb.sum())
                    pts, b = np.vstack([pts, cp]), np.r_[b, cb]
                swept.append(pts[::3])
                bad.append(b[::3])
            q = q_next
        if (wp.release or (wp.grip is not None and wp.grip >= GRIP_OPEN - 1e-6)) and holding:
            holding, placed = False, True
            stands = tuple(wp.at or plan.place)
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
    src = ", ".join(f"{k} {v}" for k, v in why.items() if v)
    print(f"   fused-view check ({label}): " + ("clear" if ok else f"{int(bad.sum())} points inside a clearance ({src})") +
          " -> captures/fused_views.jpg")
    ws.jaw_half = keep_jaw
    return ok


def _below_heightmap(pts: np.ndarray, hm: tuple) -> np.ndarray:
    """Points below the top (+1 cm) of any raised height-map cell within the margin (hm is pre-dilated)."""
    lo, cell, Hm = hm
    ix = np.floor((pts[:, 0] - lo[0]) / cell).astype(int)
    iy = np.floor((pts[:, 1] - lo[1]) / cell).astype(int)
    ok = (ix >= 0) & (ix < Hm.shape[0]) & (iy >= 0) & (iy < Hm.shape[1])
    out = np.zeros(len(pts), bool)
    top = Hm[ix[ok], iy[ok]]
    out[ok] = (top > 0.012) & (pts[ok, 2] < top + 0.01)
    return out


def safe_park(arm: Arm, planner: AzPlanner, yaw: float, tilt: float) -> None:
    """Park after an error when nothing is held: straight up to travel height first, keeping the wrist as
    the plan had it (a joint-space glide to REST from low beside an object sweeps the arm through it),
    then through READY to REST."""
    arm.set_grip(GRIP_OPEN, 1.0)
    try:
        q = np.array(arm.last_cmd[:6], dtype=float)
        here = planner.fk_pos(q)
        steps = max(1, int(np.ceil((Z_TRAVEL_MIN - here[2]) / 0.02)))
        lift = []
        for k in range(1, steps + 1) if here[2] < Z_TRAVEL_MIN else ():
            z = here[2] + (Z_TRAVEL_MIN - here[2]) * k / steps
            q, _, _ = planner.ik(here[0], here[1], z, q, yaw, tilt)
            lift.append(q)
        if lift:
            arm.glide_path(lift, 1.5)  # one smooth lift, not a stop every 2 cm
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
        if not holding:
            print(f"   could not clear the object cleanly ({e}); opening the gripper where it stands")
            arm.set_grip(GRIP_OPEN, 1.5)
        else:
            print(f"   could not set it down cleanly ({e})")
            # the pick spot is where the object itself stood a moment ago: whatever a rescan now reads there is the
            # object (or its shadow), not something to steer around. Try the way down once more without it.
            keep = ws.obstacles
            ws.obstacles = [o for o in keep if o.poly is not None or
                            float(np.hypot(*(np.array(o.xy) - np.array(plan.pick)))) > plan.obj.radius + MATCH_R]
            try:
                move_line(arm, planner, ws, (bx, by, plan.z_travel), plan.yaw, plan.tilt, 3.0)
                move_line(arm, planner, ws, (bx, by, plan.z_grasp), plan.yaw, plan.tilt, 3.0)
                settle(arm, planner, bx, by, plan.z_grasp, plan.yaw, plan.tilt)
                arm.set_grip(GRIP_OPEN, 1.5)
                move_line(arm, planner, ws, (bx, by, plan.z_travel), plan.yaw, plan.tilt, 2.5)
                print("   set it down at the pick spot (ignoring what a rescan read on that very spot)")
            except RuntimeError as e2:
                # never let go of it in the air: a filled bottle dropped from travel height spills or breaks
                print(f"!! STILL HOLDING the object and no safe way down ({e2}): TAKE IT FROM THE GRIPPER - it opens "
                      "in 30 s")
                time.sleep(30.0)
                arm.set_grip(GRIP_OPEN, 1.5)
            finally:
                ws.obstacles = keep
    safe_park(arm, planner, plan.yaw, plan.tilt)  # straight up first, never a joint-space swing past the object


# ==================================================================================== execution
GRASP_TRIES = 3  # closes on nothing -> open, look again, re-centre, close again; this many closes in all
GRASP_CENTRE_TOL = 0.008  # m: the object's centre may sit this far off the middle of the jaw before re-centring
GRASP_CENTRE_ROUNDS = 3  # look-and-shift rounds before each close
GRASP_STEP_DOWN = 0.004  # m the fingertips go lower after each close on nothing
GRASP_Z_FLOOR = 0.004  # m: never below this (the model's tips; the real ones have read ~1 cm higher)
GRASP_MAX_SHIFT = 0.025  # m: a jaw-check re-centring larger than this is not trusted


def jaw_alignment(plan: Plan, planner: AzPlanner, arm: Arm, idx: int) -> tuple[float, float, str]:
    """Where the object sits between the OPEN fingers, measured in the images - not through the calibration.

    In every calibrated camera: the two fingers are the dark blobs near where each fingertip should be; the
    object is what differs from the empty-zone background between them. The object's position along the line
    from one finger to the other, t (0 = on finger A, 1 = on finger B, 0.5 = centred), times the jaw's real
    opening, is its offset in metres - a ratio inside one image, so a camera that is off by 2 cm still
    measures it right. Views are weighted by how long the jaw looks in them. Returns (offset along the jaw
    direction in m, confidence 0..1, note); positive offset = towards finger B."""
    if not FUSED_CAMS:
        return 0.0, 0.0, "no cameras"
    q = np.array(arm.last_cmd[:6], dtype=float)
    tip = planner.fk_pos(arm.q())
    keep = planner.az
    planner.az = plan.az
    jd = np.asarray(planner.jaw_dir(plan.pick, plan.yaw, plan.tilt, plan.az), dtype=float)
    planner.az = keep
    jd = jd / (np.linalg.norm(jd) + 1e-9)
    half = plan.open_grip * JAW_STROKE / 2
    zc = max(0.005, min(plan.z_grasp, plan.obj.height) / 2)  # look at the object's lower half
    A = np.array([tip[0], tip[1], zc]) - half * jd
    B = np.array([tip[0], tip[1], zc]) + half * jd
    num, den, notes, tiles = 0.0, 0.0, [], []
    for k, lc in FUSED_CAMS.items():
        if k not in FUSED_MODELS:
            continue
        cam, f = FUSED_MODELS[k], lc.latest()
        bg = cv2.imread(str(BACKGROUND[k]))
        if f is None or bg is None:
            continue
        a, b = cam.project(np.vstack([A, B]))
        L = float(np.linalg.norm(b - a))
        if not np.all(np.isfinite([*a, *b])) or L < 25:
            continue  # the jaw is end-on or off-frame in this view
        h, w = f.shape[:2]
        dark = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[..., 2] < 70
        # the finger plates are wide solid blobs; a cable crossing the view is a thin line: open it away
        dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((11, 11), np.uint8)) > 0
        def blob(c):
            m = np.zeros((h, w), bool)
            cv2.circle(m.view(np.uint8), (int(c[0]), int(c[1])), int(max(12, 0.35 * L)), 1, -1)
            ys, xs = np.nonzero(m & dark)
            return None if len(xs) < 40 else np.array([xs.mean(), ys.mean()])
        fa, fb = blob(a), blob(b)
        if fa is None or fb is None:
            notes.append(f"{k}: fingers not both visible")
            continue
        band = np.zeros((h, w), np.uint8)
        cv2.line(band, (int(fa[0]), int(fa[1])), (int(fb[0]), int(fb[1])), 255, int(max(10, 0.5 * L)))
        diff = table_foreground(f, bg, quiet=True) > 0  # the object, not its shadow on the sheet
        obj = (band > 0) & diff & ~cv2.dilate(dark.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        ys, xs = np.nonzero(obj)
        if len(xs) < 60:
            notes.append(f"{k}: object not visible between the fingers")
            continue
        o = np.array([xs.mean(), ys.mean()])
        ab = fb - fa
        t = float((o - fa) @ ab / (ab @ ab))
        off = (t - 0.5) * 2 * half
        wgt = L
        num, den = num + wgt * off, den + wgt
        notes.append(f"{k}: object at {t:.2f} of the jaw ({off * 100:+.1f} cm)")
        v = f.copy()
        v[obj] = (0.5 * v[obj] + [0, 90, 0]).astype(np.uint8)
        for c, col in ((fa, (255, 0, 0)), (fb, (0, 0, 255)), (o, (0, 255, 0))):
            cv2.drawMarker(v, (int(c[0]), int(c[1])), col, cv2.MARKER_CROSS, 24, 2)
        cu, cv_ = int((fa[0] + fb[0]) / 2), int((fa[1] + fb[1]) / 2)
        r = int(max(80, 1.3 * L))
        tiles.append(cv2.resize(v[max(0, cv_ - r):cv_ + r, max(0, cu - r):cu + r], (320, 320)))
    if tiles:
        cv2.imwrite(str(CAPTURES / f"grasp_check_{idx}.jpg"), np.hstack(tiles))
    if den == 0:
        return 0.0, 0.0, "; ".join(notes) or "no usable view"
    return num / den, min(1.0, den / 200.0), "; ".join(notes)


def centre_in_jaw(plan: Plan, planner: AzPlanner, arm: Arm, ws: "Workspace", tag: str) -> None:
    """Before closing: measure the object between the open fingers (jaw_alignment) and, while it is more than
    GRASP_CENTRE_TOL off the middle, lift 3 cm, shift along the jaw by the measured offset, and come back down."""
    keep = planner.az
    planner.az = plan.az
    jd = np.asarray(planner.jaw_dir(plan.pick, plan.yaw, plan.tilt, plan.az), dtype=float)
    planner.az = keep
    jd = jd / (np.linalg.norm(jd[:2]) + 1e-9)
    for r in range(GRASP_CENTRE_ROUNDS):
        time.sleep(0.4)  # let the cameras see the arm at rest
        off, conf, note = jaw_alignment(plan, planner, arm, f"{tag}_{r}")
        print(f"   jaw check: {note} -> {off * 100:+.1f} cm off centre (confidence {conf:.1f}) "
              f"-> captures/grasp_check_{tag}_{r}.jpg")
        if conf < 0.5 or abs(off) <= GRASP_CENTRE_TOL:
            return
        if abs(off) > GRASP_MAX_SHIFT:
            print(f"   (not re-centring on a {off * 100:.1f} cm reading: larger than the check is trusted for)")
            return
        here = planner.fk_pos(np.array(arm.last_cmd[:6], dtype=float))
        up = here + np.array([0, 0, 0.03])
        to = up + np.r_[off * jd[:2], 0.0]
        print(f"   re-centring: lift, shift {off * 100:+.1f} cm along the jaw, come back down")
        move_line(arm, planner, ws, up, plan.yaw, plan.tilt, 1.2)
        move_line(arm, planner, ws, to, plan.yaw, plan.tilt, 1.2)
        move_line(arm, planner, ws, to - np.array([0, 0, 0.03]), plan.yaw, plan.tilt, 1.2)


TOUCH_OFFSETS = (0.0, 0.015, -0.015, 0.03, -0.03)  # m along the jaw: where the touch search tries to come down
TOUCH_STEP = 0.003  # m per step of the probing descent
TOUCH_BLOCK = 0.006  # m: settled this far above the commanded height = a finger standing on something


def probe_descend(arm: Arm, planner: AzPlanner, xy, z_from: float, z_to: float, yaw: float, tilt: float) -> tuple[bool, float]:
    """One smooth descent to the grasp height, then read where the fingertips really are. Small probing steps
    did not work: a 3 mm step is below what the joints move from standstill, so the arm stayed put and read
    as a block at ~3 cm, every time. A finger standing on the object keeps the arm clearly above its command
    once it has settled (7 mm on a dish, seen in the recording) while a free descent settles to within a few mm.
    Returns (reached the target height, measured fingertip height)."""
    q = np.array(arm.last_cmd[:6], dtype=float)
    path = []
    for z in np.linspace(z_from, z_to, max(2, int((z_from - z_to) / 0.01) + 1))[1:]:
        q, _, _ = planner.ik(xy[0], xy[1], z, q, yaw, tilt)
        path.append(q)
    arm.glide_path(path, 1.5)
    time.sleep(0.8)
    zm = planner.fk_pos(arm.q())[2]
    return zm <= z_to + TOUCH_BLOCK, float(zm)


def touch_search(plan: Plan, planner: AzPlanner, arm: Arm) -> bool:
    """Find, by touch, where the open fingers come down BESIDE the object instead of on it. The cameras place
    a low object to a centimetre or two, the jaw may have millimetres to spare (a 9.0 cm dish in a 9.5 cm jaw):
    a finger landing on it is felt as the arm stopping short, and the next spot along the jaw is tried."""
    keep = planner.az
    planner.az = plan.az
    jd = np.asarray(planner.jaw_dir(plan.pick, plan.yaw, plan.tilt, plan.az), dtype=float)
    planner.az = keep
    jd = jd[:2] / (np.linalg.norm(jd[:2]) + 1e-9)
    here = planner.fk_pos(np.array(arm.last_cmd[:6], dtype=float))
    z_hover = max(here[2], plan.z_grasp + 0.03)
    for off in TOUCH_OFFSETS:
        xy = np.array(here[:2]) + off * jd
        q, _, _ = planner.ik(xy[0], xy[1], z_hover, np.array(arm.last_cmd[:6], dtype=float), plan.yaw, plan.tilt)
        arm.glide(q, 1.0)
        ok, zm = probe_descend(arm, planner, xy, z_hover, plan.z_grasp, plan.yaw, plan.tilt)
        print(f"   touch search {off * 100:+.1f} cm along the jaw: " +
              (f"reached {zm * 100:.1f} cm - the fingers are beside it" if ok else
               f"stopped at {zm * 100:.1f} cm (commanded lower): a finger is on it, lifting"))
        if ok:
            return True
        q, _, _ = planner.ik(xy[0], xy[1], z_hover, np.array(arm.last_cmd[:6], dtype=float), plan.yaw, plan.tilt)
        arm.glide(q, 0.8)
    return False


BRUTE_PRIOR = (-0.010, -0.014)  # m: the visual check's consistent correction on the dish runs: the grid's centre
BRUTE_ALONG = 0.005  # m grid step along the jaw (where the object may have mm to spare)
BRUTE_ACROSS = 0.010  # m grid step across the jaw (the fingers are wide there)
BRUTE_RANGE = (0.040, 0.020)  # m half-extent along / across the jaw


def brute_candidates(plan: Plan, planner: AzPlanner, centre) -> list[np.ndarray]:
    """Every grasp spot on a jaw-aligned grid around `centre`, nearest first (across-the-jaw distance counts
    half: the fingers are wide there, so an error across the jaw costs less)."""
    keep = planner.az
    planner.az = plan.az
    jd = np.asarray(planner.jaw_dir(plan.pick, plan.yaw, plan.tilt, plan.az), dtype=float)[:2]
    planner.az = keep
    jd = jd / (np.linalg.norm(jd) + 1e-9)
    pd = np.array([-jd[1], jd[0]])
    pts = []
    for a in np.arange(-BRUTE_RANGE[0], BRUTE_RANGE[0] + 1e-9, BRUTE_ALONG):
        for b in np.arange(-BRUTE_RANGE[1], BRUTE_RANGE[1] + 1e-9, BRUTE_ACROSS):
            pts.append((np.hypot(a, 0.5 * b), np.asarray(centre) + a * jd + b * pd))
    return [p for _, p in sorted(pts, key=lambda t: t[0])]


DISC_FULL = 0.008  # m: a round object is held across its diameter once the jaw is within this of it


def centre_on_disc(plan: Plan, planner: AzPlanner, arm: Arm, args: "Args", xy: np.ndarray, z_h: float) -> bool:
    """A round object held across a chord - the jaw closed at w < its diameter - sits sqrt(r^2 - (w/2)^2)
    off-centre, perpendicular to the jaw, and tips as it is lifted. Re-grasp shifted by that much: one side
    first (then once more that way if the chord grew but is not full yet), else the other side. Ends on the
    widest grasp found, so the jaw spans the diameter and the object is carried level."""
    r = plan.obj.diameter / 2
    jaw = planner.jaw_dir(tuple(xy), plan.yaw, plan.tilt, plan.az)[:2]
    u = np.array([-jaw[1], jaw[0]]) / (np.linalg.norm(jaw) + 1e-9)
    full = lambda w: w >= 2 * r - DISC_FULL  # noqa: E731
    offc = lambda w: float(np.sqrt(max(r * r - (w / 2) ** 2, 0.0)))  # noqa: E731
    at = np.array(xy, dtype=float)  # where the gripper is now

    def regrasp(to: np.ndarray) -> float | None:
        nonlocal at
        try:  # open wide (a dish can be 5 mm narrower than the stroke), rise straight up, then shift
            arm.set_grip(1.0, 0.8)
            time.sleep(0.3)
            q, _, _ = planner.ik(at[0], at[1], z_h, np.array(arm.last_cmd[:6], dtype=float), plan.yaw, plan.tilt)
            arm.glide(q, 1.0)
            q, _, _ = planner.ik(to[0], to[1], z_h, q, plan.yaw, plan.tilt)
            arm.glide(q, 0.8)
            path, qq = [], q
            for z in np.linspace(z_h, plan.z_grasp, 4)[1:]:
                qq, _, _ = planner.ik(to[0], to[1], z, qq, plan.yaw, plan.tilt)
                path.append(qq)
            arm.glide_path(path, 1.6)  # slower: a finger landing on the rim should not push it
            time.sleep(0.3)
        except RuntimeError as e:
            print(f"   (re-grasp at ({to[0]:+.3f}, {to[1]:+.3f}) unreachable: {e})")
            return None
        at = np.array(to, dtype=float)
        held = grip_on_object(arm, args, plan.obj.grip_width)
        w = float(arm.state7()[6]) * JAW_STROKE if held else 0.0
        print(f"   re-grasp at ({to[0]:+.3f}, {to[1]:+.3f}): across {w * 100:.1f} cm")
        return w

    w0 = float(arm.state7()[6]) * JAW_STROKE
    if full(w0):
        print(f"   held across {w0 * 100:.1f} cm of its {2 * r * 100:.1f} cm diameter: centred")
        return True
    print(f"   held across a {w0 * 100:.1f} cm chord of a {2 * r * 100:.1f} cm disc: "
          f"{offc(w0) * 100:.1f} cm off-centre - re-grasping centred")
    tried = [(w0, np.array(xy, dtype=float))]
    wa = regrasp(xy + offc(w0) * u)
    if wa is not None:
        tried.append((wa, at.copy()))
        if full(wa):
            return True
        if wa > w0 + 0.004:  # the right way, not far enough yet
            wb = regrasp(at + offc(wa) * u)
            if wb is not None:
                tried.append((wb, at.copy()))
                if full(wb):
                    return True
    if max(t[0] for t in tried) <= w0 + 0.004:  # the other way
        wc = regrasp(xy - offc(w0) * u)
        if wc is not None:
            tried.append((wc, at.copy()))
            if full(wc):
                return True
    w_best, xy_best = max(tried, key=lambda t: t[0])
    w = float(arm.state7()[6]) * JAW_STROKE
    if not np.allclose(xy_best, at):
        w = regrasp(xy_best) or 0.0
        print(f"   back to the widest grasp ({w_best * 100:.1f} cm): now across {w * 100:.1f} cm")
    if w < 0.5 * w_best:  # it is not where it was held: it was nudged - the caller searches again
        print("   !! the object is no longer where it was held (nudged while re-grasping)")
        return False
    return True


def brute_grasp(plan: Plan, planner: AzPlanner, arm: Arm, args: "Args", live=None) -> bool:
    """Try grasp spots until one holds: hover, one smooth descent to the grasp height, close. Holding -> done
    (the plan carries on from here). Closed on nothing -> open, rise, next spot. After a full grid, the object
    is measured again (it may have been nudged) and one more round is tried."""
    z_h = plan.z_grasp + 0.03
    centre = np.array(plan.pick) + np.array(BRUTE_PRIOR)
    n = 0
    for rnd in range(2):
        cands = brute_candidates(plan, planner, centre)
        print(f"   brute force round {rnd + 1}: {len(cands)} spots around ({centre[0]:+.3f}, {centre[1]:+.3f}), nearest first")
        for xy in cands:
            n += 1
            if live is not None:
                live.set(phase=f"brute-force grasp: try {n} at ({xy[0]:+.3f}, {xy[1]:+.3f})")
            try:
                q, _, _ = planner.ik(xy[0], xy[1], z_h, np.array(arm.last_cmd[:6], dtype=float), plan.yaw, plan.tilt)
                arm.glide(q, 0.9)
                path, qq = [], q
                for z in np.linspace(z_h, plan.z_grasp, 4)[1:]:
                    qq, _, _ = planner.ik(xy[0], xy[1], z, qq, plan.yaw, plan.tilt)
                    path.append(qq)
                arm.glide_path(path, 1.0)
                time.sleep(0.3)
            except RuntimeError as e:
                print(f"   try {n}: unreachable ({e})")
                continue
            if grip_on_object(arm, args, plan.obj.grip_width):
                print(f"   try {n} at ({xy[0]:+.3f}, {xy[1]:+.3f}): HOLDING - "
                      f"{np.linalg.norm(xy - np.array(plan.pick)) * 100:.1f} cm from the camera estimate")
                if plan.obj.long_axis is None and plan.obj.measured:
                    if not centre_on_disc(plan, planner, arm, args, np.array(xy, dtype=float), z_h):
                        arm.set_grip(1.0, 0.8)
                        q, _, _ = planner.ik(xy[0], xy[1], z_h, np.array(arm.last_cmd[:6], dtype=float), plan.yaw, plan.tilt)
                        arm.glide(q, 0.8)
                        continue  # keep searching (nearest first from the last hold)
                return True
            print(f"   try {n} at ({xy[0]:+.3f}, {xy[1]:+.3f}): nothing - next")
            arm.set_grip(plan.open_grip, 0.8)
            q, _, _ = planner.ik(xy[0], xy[1], z_h, np.array(arm.last_cmd[:6], dtype=float), plan.yaw, plan.tilt)
            arm.glide(q, 0.6)
        if rnd == 0 and args.given:
            print("   full grid tried: measuring the object again for a second round")
            try:  # rise clear first so the camera sees it
                here = planner.fk_pos(np.array(arm.last_cmd[:6], dtype=float))
                q, _, _ = planner.ik(here[0], here[1], plan.z_travel, np.array(arm.last_cmd[:6], dtype=float), plan.yaw, plan.tilt)
                arm.glide(q, 1.5)
                time.sleep(0.8)
                new = measure_disc(plan.obj.diameter / 2, plan.obj.height)
                if new is not None:
                    print(f"   the object is now at ({new[0]:+.3f}, {new[1]:+.3f})")
                    centre = np.array(new) + np.array(BRUTE_PRIOR)
            except Exception as e:
                print(f"   (could not re-measure: {e})")
    return False


def measure_disc(r: float, h: float) -> tuple[float, float] | None:
    """A flat round object's centre from the detection camera: fit a cylinder of radius r, height h to its
    outline against the empty-zone background (rays touch the rim between the table and its top)."""
    from scipy.optimize import least_squares
    lc = FUSED_CAMS.get(DET)
    if lc is None:
        return None
    c, f = FUSED_MODELS[DET], lc.latest()
    bg = cv2.imread(str(BACKGROUND[DET]))
    Z = load_zone()
    zm = np.zeros(f.shape[:2], np.uint8)
    cv2.fillPoly(zm, [c.project(np.c_[Z, np.zeros(len(Z))]).astype(np.int32)], 255)
    d = cv2.absdiff(cv2.GaussianBlur(f, (5, 5), 0), cv2.GaussianBlur(bg, (5, 5), 0)).max(axis=2)
    m = ((d > 14) & (zm > 0)).astype(np.uint8) * 255
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if not cnts:
        return None
    pts = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)[::2]
    rays = [c.ray(u, v) for u, v in pts]
    def resid(pp):
        out = []
        for o, dv in rays:
            zs = np.linspace(0, h, 7); sc = (zs - o[2]) / dv[2]; P = o[None] + sc[:, None] * dv[None]
            out.append(np.min(np.abs(np.hypot(P[:, 0] - pp[0], P[:, 1] - pp[1]) - r)))
        return np.array(out)
    sol = least_squares(resid, np.mean([c.hit_plane(u, v, h / 2)[:2] for u, v in pts], axis=0), loss="soft_l1", f_scale=0.003)
    return float(sol.x[0]), float(sol.x[1])


SKIN = None  # the tactile skin while a --tactile run is going (soft grasps use it to feel contact and slip)


def site_rot(planner: AzPlanner, q6: np.ndarray) -> np.ndarray:
    """World rotation of the grasp site (the gripper frame) at joints q6."""
    m, d = planner.model, planner.data
    d.qpos[:] = 0
    d.qpos[:6] = q6
    mujoco.mj_kinematics(m, d)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    return d.site_xmat[sid].reshape(3, 3).copy()


def _axis_rot(axis: np.ndarray, th: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def pour_path(planner: AzPlanner, ws: "Workspace", q_start: np.ndarray, open_dir: np.ndarray, reach: float,
              over: np.ndarray, lip_z: float, tilt_max: float, dish_r: float) -> tuple[list, int] | None:
    """Joint path for pouring from a held tube: its axis runs `reach` m from the grasp site to the open end,
    along `open_dir` (world, at the grasp). Level first with the lip over `over` at `lip_z`, then rotate the
    gripper about its jaw axis so the open end tips down while the lip stays put (tilt_max rad), then back
    to level. Tries wrist turns about the vertical until IK and the table/dish checks pass. Returns
    (path, index of the full tilt) or None."""
    R0 = site_rot(planner, q_start)
    v_loc = R0.T @ (open_dir / np.linalg.norm(open_dir))
    jaw = planner.jaw_local
    lip = np.array([over[0], over[1], lip_z])
    for phi in (0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 2.6, -2.6, 3.14):
        Rl = _axis_rot(np.array([0, 0, 1.0]), phi) @ R0
        # the tilt sign that brings the open end DOWN
        sgn = 1.0 if (Rl @ _axis_rot(jaw, 0.5) @ v_loc)[2] < 0 else -1.0
        path, q, ok = [], np.array(q_start[:6], dtype=float), True
        ths = list(np.linspace(0, tilt_max, 10)) + list(np.linspace(tilt_max, 0, 6))[1:]
        for th in ths:
            R = Rl @ _axis_rot(jaw, sgn * th)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = lip - R @ v_loc * reach
            q0 = np.zeros(planner.model.nq)
            q0[:6] = q
            good, qq = planner.kin.ik(T, "grasp_site", init_q=q0, limits=planner.limits, max_iters=400,
                                      pos_threshold=2e-3, ori_threshold=3e-2)
            if not good:
                ok = False
                break
            qq = np.array(qq[:6])
            if path and np.max(np.abs(qq - path[-1])) > 0.6:  # IK jumped branch mid-pour
                ok = False
                break
            pts = ws.points(qq)
            near = np.linalg.norm(pts[:, :2] - over[:2], axis=1) < dish_r + 0.015
            if pts[:, 2].min() < 0.015 or (near.any() and pts[near, 2].min() < 0.035):
                ok = False
                break
            path.append(qq)
            q = qq
        if ok:
            # the move into the pour from where the arm is, swept too
            gap, who, zmin = ws.scan(np.array(q_start[:6]), path[0])
            if zmin < 0.015:
                continue
            print(f"   pour: wrist turned {np.degrees(phi):+.0f} deg, tilt to {np.degrees(tilt_max):.0f} deg, "
                  f"lip held at ({lip[0]:+.3f}, {lip[1]:+.3f}, {lip[2]:.3f})")
            return path, 9
    return None


def do_pour(arm: Arm, planner: AzPlanner, ws: "Workspace", plan: Plan, args: "Args") -> None:
    """--pour-over: pour the held tube into the dish, then come back to the lift pose so the plan carries on."""
    over = np.array([float(v) for v in args.pour_over.split(",")[:2]])
    tube_open = np.array([float(v) for v in args.pour_open_end.split(",")[:2]])
    d = tube_open - np.asarray(plan.pick)
    open_dir = np.r_[d, 0.0]
    reach = float(np.linalg.norm(d))
    q_lift = np.array(arm.last_cmd[:6], dtype=float)
    got = pour_path(planner, ws, q_lift, open_dir, reach, over, args.pour_lip_z, np.radians(args.pour_tilt), 0.045)
    if got is None:
        print("   !! pour: no reachable, collision-free pouring pose - skipping the pour")
        return
    path, full = got
    arm.glide_path([path[0]], 3.0)  # level, lip over the dish
    arm.glide_path(path[1:full + 1], 3.5)  # tip it slowly
    print(f"   pouring: holding {args.pour_hold:.1f} s")
    time.sleep(args.pour_hold)
    arm.glide_path(path[full + 1:], 2.5)  # back to level
    arm.glide_path([q_lift], 3.0)  # back to the lift pose


HOLD_TEST_LIFT = 0.02  # m: the hold test lift (5 mm passed a dish that slid out 2 s into the carry)
HOLD_SPEED = 1.0  # --soft: the arm never moves faster than this while it holds the object (a 2x lift dropped the dish)


def joint_effort(arm: Arm, n: int = 25, dt: float = 0.012) -> np.ndarray:
    """The six arm joints' measured torques, averaged over n reads (single reads jitter)."""
    v = []
    for _ in range(n):
        v.append(np.asarray(arm.robot.get_observations()["joint_eff"][:6], dtype=float))
        time.sleep(dt)
    return np.mean(v, axis=0)


def _lift_pose(planner: AzPlanner, plan: Plan, cmd: np.ndarray) -> np.ndarray:
    here = planner.fk_pos(cmd)
    return planner.ik(here[0], here[1], here[2] + HOLD_TEST_LIFT, cmd, plan.yaw, plan.tilt)[0]


def empty_lift(arm: Arm, planner: AzPlanner, plan: Plan) -> None:
    """The arm-torque tare for the hold test: with the jaw still open around the object, go up HOLD_TEST_LIFT and
    back, reading the joint torques at both ends. The same lift with the object held then differs only by the
    object's weight - the arm's own gravity, the controller's model error and the joints' static friction cancel."""
    keep, planner.az = planner.az, plan.az
    try:
        cmd = np.array(arm.last_cmd[:6], dtype=float)
        q_up = _lift_pose(planner, plan, cmd)
        time.sleep(0.4)
        e_down = joint_effort(arm)
        arm.glide(q_up, 1.0)
        time.sleep(0.8)
        e_up = joint_effort(arm)
        arm.glide(cmd, 0.7)
        time.sleep(0.3)
        arm.effort_tare = (e_up - e_down, q_up, cmd)
        print(f"   arm-torque tare (empty jaw, {HOLD_TEST_LIFT * 100:.0f} cm lift): d tau j2 {e_up[1] - e_down[1]:+.3f}, "
              f"j3 {e_up[2] - e_down[2]:+.3f} Nm")
    except RuntimeError as e:
        arm.effort_tare = None
        print(f"   (no arm-torque tare: {e})")
    finally:
        planner.az = keep


def felt_mass(arm: Arm, planner: AzPlanner, d_held: np.ndarray) -> float | None:
    """kg hanging at the fingertips: the torque change of a held lift minus the empty-jaw tare, projected on the
    torques a unit weight at the grasp site needs (shoulder and elbow carry it). The measured torques' sign
    convention is taken from the tare itself, which must follow the model's own gravity change."""
    tare = getattr(arm, "effort_tare", None)
    if tare is None:
        return None
    d_empty, q_up, q_down = tare
    m, d = planner.model, planner.data

    def bias(q):
        d.qpos[:] = 0
        d.qpos[:6] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        return d.qfrc_bias[:6].copy()

    dg = bias(q_up) - bias(q_down)  # the model's required-torque change for the empty lift
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    d.qpos[:] = 0
    d.qpos[:6] = q_up
    mujoco.mj_forward(m, d)
    J = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, J, None, sid)
    per_kg = J[:, :6].T @ np.array([0.0, 0.0, 9.81])  # required torque per kg held, same convention as dg
    num = den = 0.0
    for j in (1, 2):
        sgn = np.sign(d_empty[j] * dg[j]) if abs(dg[j]) > 0.05 and abs(d_empty[j]) > 0.02 else 1.0
        num += sgn * (d_held[j] - d_empty[j]) * per_kg[j]
        den += per_kg[j] ** 2
    return float(num / den) if den > 1e-9 else None


def secure_hold(arm: Arm, planner: AzPlanner, plan: Plan, args: "Args") -> bool:
    """False when the grasp is lost (the jaw is empty): the caller stops instead of carrying nothing."""
    arm.speed_free = getattr(arm, "speed", 1.0)  # restored at the release
    arm.speed = min(arm.speed_free, HOLD_SPEED)
    return _secure_hold(arm, planner, plan, args) is not False


def _secure_hold(arm: Arm, planner: AzPlanner, plan: Plan, args: "Args") -> bool | None:
    """After a soft close: lift 5 mm and read the gripper motor load. Still loaded (the object's weight and the
    hold are on the fingers) -> carry on. Load gone, or the skin lost most of its pressure -> it is slipping:
    set it back down and raise the held load by 0.05 (never past 3x the first target)."""
    from pick_bottle import load
    free = getattr(arm, "free_load", 0.0)
    target = getattr(arm, "hold_load", free + args.grip_load)
    keep, planner.az = planner.az, plan.az
    try:
        cmd = np.array(arm.last_cmd[:6], dtype=float)
        q_up = _lift_pose(planner, plan, cmd)
        skin_first = None
        for k in range(5):
            s_held = SKIN.magnitude() if SKIN is not None else None
            skin_first = skin_first if skin_first is not None else s_held
            e_down = joint_effort(arm)
            arm.glide(q_up, 1.0)
            time.sleep(0.8)  # a slow slip shows within a second
            e = load(arm, 8)
            e_up = joint_effort(arm)
            s_up = SKIN.magnitude() if SKIN is not None else None
            kg = felt_mass(arm, planner, e_up - e_down)
            # three witnesses: the gripper motor still loaded, the skin still pressed, and the ARM carrying the
            # weight - the last catches an object sliding down the fingers while its base stays on the table
            light = kg is not None and args.min_mass is not None and kg < args.min_mass
            slip = (e < free + 0.5 * (target - free) or (s_held is not None and s_held > 100 and s_up < 0.5 * s_held)
                    or light)
            sk = "" if s_held is None else f", skin {s_held:.0f} -> {s_up:.0f}"
            ms = "" if kg is None else f", arm feels {kg * 1000:.0f} g"
            print(f"   hold test {k + 1}: load {e:.2f} (free {free:.2f}, target {target:.2f}){sk}{ms}: "
                  + ("slipping - back down, firmer" if slip else "held")
                  + (f" (under --min-mass {args.min_mass * 1000:.0f} g: it is not lifting it)" if light else ""))
            if not slip:
                arm.skin_held = s_up
                arm.felt_kg = kg
                return
            if e < free + 0.03 and (s_up is None or s_up < 0.8 * (s_held or 1e9) or k > 0):
                print("   !! the jaw is empty (load at its free level): the grasp was lost - not squeezing harder")
                arm.glide(cmd, 0.7)
                return False
            arm.glide(cmd, 0.7)
            cap = args.skin_max if args.skin_max is not None else (3.0 * skin_first if skin_first and skin_first > 50 else None)
            if SKIN is not None and cap is not None and SKIN.magnitude() >= cap:
                print(f"   !! the skin reads {SKIN.magnitude():.0f} (cap {cap:.0f}): squeezing harder would crush or squirt it "
                      "- not firming; it stays on the table")
                return False
            # never near the load that flexed the dish (or, for a squeezable bottle, the one that squirts it)
            target = min(target + 0.05, free + (args.grip_load_max if args.grip_load_max is not None else args.grip_load + 0.10))
            for _ in range(30):  # close in small steps until the load reaches the new target
                if load(arm, 6) >= target:
                    break
                arm.grip = max(0.0, arm.grip - 0.001)
                arm._cmd(cmd)
                time.sleep(0.05)
        # every hold test failed: it was set back down after each one - leave it there, never carry a slipping grip
        print("   !! no hold test passed: leaving it on the table")
        return False
    finally:
        planner.az = keep


def grip_on_object(arm: Arm, args: "Args", width: float | None = None) -> bool:
    """Close until the fingers stall on the object. True if something is actually held. --soft: stop at the
    first touch (opening lag, motor effort or the skin) and hold with a ~0.5 mm preload."""
    if args.soft:
        from pick_bottle import soft_close
        start = None if width is None else min(arm.grip, (width + 0.01) / JAW_STROKE)
        g = soft_close(arm, start=start, hold_load=args.grip_load, skin=SKIN)
    else:
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
    if not FAST and not view_gate(plan, planner, ws, arm.q(), frames, "planned trajectory"):
        print("!! the fused-view check found the arm inside a clearance: not moving")
        return False
    if args.servo and not plan.servo_ok and not args.open_loop:
        print(f"!! no safe spot for the visual check ({plan.servo_note}): not running open-loop - the arm's sag "
              "offset makes the fingers miss. Move the object, or pass --open-loop to accept that.")
        return False
    unfold, above = plan.waypoints[0], plan.waypoints[1]
    step(0, unfold.label)
    arm.set_grip(unfold.grip if unfold.grip is not None else GRIP_OPEN, 1.0)
    q = above.qs[-1] if above.qs else planner.ik(above.xyz[0], above.xyz[1], above.xyz[2], unfold.q, plan.yaw,
                                                  plan.tilt)[0]
    if args.blend and not (args.servo and plan.servo_ok and cam is not None):
        step(1, above.label)
        print(f"   (blended: {unfold.label} -> {above.label})")
        smooth_follow(arm, planner, ws, [np.array(unfold.q), np.asarray(q, dtype=float)],
                      unfold.seconds + above.seconds, None, 0.01)
    else:
        goto_joint(arm, planner, np.array(unfold.q), unfold.seconds)
        step(1, above.label)
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
        if not FAST and not view_gate(plan, planner, ws, arm.q(), frames, "corrected trajectory", plan.lead + plan.waypoints[2:]):
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


_FINDERS: dict = {}


def _finder(path: str):
    """A --refind module: a file exposing find(frame, cam, guess_xy) -> (x, y) or None."""
    if path not in _FINDERS:
        import importlib.util

        spec = importlib.util.spec_from_file_location(Path(path).stem, str(HERE / path if not Path(path).is_absolute() else path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _FINDERS[path] = mod
    return _FINDERS[path]


REAIM_STILL = 0.004  # m: the object moved less than this - slide back in as planned


def reaim_target(plan: Plan, planner: AzPlanner, arm: Arm, ws: "Workspace", args: "Args", idx: int,
                 cam: LiveCamera | None, cmodel: CameraModel) -> Plan | None:
    """Between close attempts (--refind): open, back the fingers out to this grasp's pre-pose, re-measure the
    object on the detection camera, and re-aim the grasp at where it is now (the rest of the plan rebuilt and
    re-validated from here). None when it is not found, moved more than --refind-max, or has no safe path."""
    arm.set_grip(GRIP_OPEN if plan.mode == "side" else plan.open_grip, 1.2)
    pre, slide = plan.waypoints[idx - 2], plan.waypoints[idx - 1]  # pre-grasp (or above-grasp), slide (or descend)
    move_line(arm, planner, ws, pre.xyz, plan.yaw, plan.tilt, 2.0)
    time.sleep(0.8)  # let the frame catch up with the arm standing still
    frame = cam.latest() if cam is not None else None
    if frame is None:
        print("   (no detection camera frame to re-measure on)")
        return None
    off = np.array([float(v) for v in args.refind_offset.split(",")])
    guess = np.array(plan.pick) - off  # where the camera should see it
    try:
        got = _finder(args.refind).find(frame, cmodel, guess)
    except Exception as e:
        print(f"   (re-measure failed: {e})")
        got = None
    if got is None:
        print("   !! re-measure: the object was not found near where it was")
        return None
    new = np.asarray(got, dtype=float)[:2] + off
    d = float(np.linalg.norm(new - np.array(plan.pick)))
    print(f"   re-measured: now at ({got[0]:+.3f}, {got[1]:+.3f}) - {d * 100:.1f} cm from where the grasp aimed")
    if d > args.refind_max:
        print(f"   !! it moved more than --refind-max {args.refind_max * 100:.0f} cm: not chasing it")
        return None
    if d < REAIM_STILL:
        alt = plan
    else:
        alt = replan(plan, planner, tuple(float(v) for v in new), args.hold, args.pause if args.rescan else 0.0)
        alt.obj.xy = tuple(float(v) for v in new)
        gap, who, legs, _ = validate(planner, ws, alt, arm.q(), wps=alt.waypoints[idx - 2:])
        if gap < 0:
            print(f"   !! no safe path to it there: {abs(gap) * 100:.1f} cm inside {who}")
            return None
        alt.legs = legs
        print(f"   re-aimed: clearance {fmt_gap(gap)}" + (f" from {who}" if who else ""))
    ws.aim(alt.obj, alt.pick)
    for wp in alt.waypoints[idx - 2: idx]:  # to the (new) pre-grasp, then in around it again
        move_line(arm, planner, ws, wp.xyz, alt.yaw, alt.tilt, wp.seconds, qs=wp.qs if alt is not plan else None)
        if wp.settle:
            try:
                settle(arm, planner, wp.xyz[0], wp.xyz[1], wp.xyz[2], alt.yaw, alt.tilt)
            except RuntimeError as e:
                print(f"   (sag compensation skipped: {e})")
    arm.set_grip(alt.open_grip, 0.8)
    return alt


def bottle_tilt_path(planner: AzPlanner, ws: "Workspace", q_level: np.ndarray, axis_base: np.ndarray,
                     tip: np.ndarray, lip: np.ndarray, tilt: float, bowl: tuple | None, n: int = 12) -> list | None:
    """Joint path that tilts the held squeeze bottle toward the bowl: about the horizontal axis square to its spout
    (so the spout tips DOWN), while its tip moves from where it is (`tip`, the bottle level) to `lip` - over the
    bowl, just above the rim. The arm and the bottle's body (a 7.85 cm cylinder from `axis_base` up 18 cm) are
    checked at every step against the table, the bowl (x, y, radius, rim z) and the weigh station. None if no IK."""
    R0 = site_rot(planner, q_level)
    site0 = planner.fk_pos(q_level)
    v_tip = R0.T @ (tip - site0)
    # the bottle body as points in the gripper frame (it does not move in the jaw)
    th = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    body = np.array([[axis_base[0] + 0.039 * np.cos(a), axis_base[1] + 0.039 * np.sin(a), axis_base[2] + h]
                     for h in np.linspace(0.0, 0.18, 5) for a in th])
    body_loc = (body - site0) @ R0  # rows: R0.T @ (p - site0)
    d = np.r_[tip[:2] - axis_base[:2], 0.0]
    d /= np.linalg.norm(d) + 1e-9
    ax = np.cross([0.0, 0.0, 1.0], d)  # horizontal, square to the spout
    sgn = 1.0 if (_axis_rot(ax, 0.3) @ (tip - axis_base))[2] < (tip - axis_base)[2] else -1.0
    st = station_obj()
    path, q = [], np.array(q_level[:6], dtype=float)
    for k in range(0, n + 1):  # k = 0: the level pose it starts from is checked too
        f = k / n
        R = _axis_rot(ax, sgn * tilt * f) @ R0
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = (tip + (lip - tip) * f) - R @ v_tip
        q0 = np.zeros(planner.model.nq)
        q0[:6] = q
        good, qq = planner.kin.ik(T, "grasp_site", init_q=q0, limits=planner.limits, max_iters=400,
                                  pos_threshold=2e-3, ori_threshold=3e-2)
        if not good:
            print(f"   (bottle tilt: no IK at {np.degrees(tilt * f):.0f} deg)")
            return None
        qq = np.array(qq[:6])
        if np.max(np.abs(qq - q)) > 0.6:
            print("   (bottle tilt: IK jumped branch)")
            return None
        arm_pts = ws.points(qq)
        bot = T[:3, 3] + body_loc @ R.T
        allp = np.vstack([arm_pts, bot])
        if allp[:, 2].min() < 0.015:
            print(f"   (bottle tilt: {'the bottle' if bot[:, 2].min() < 0.015 else 'the arm'} would reach the table)")
            return None
        if bowl is not None:
            bx, by, br, bz = bowl
            inside = np.linalg.norm(allp[:, :2] - [bx, by], axis=1) < br + 0.01
            if inside.any() and allp[inside, 2].min() < bz + 0.01:
                print(f"   (bottle tilt: would dip into the bowl's rim at {np.degrees(tilt * f):.0f} deg)")
                return None
        if st is not None and box_hits(allp, st, 0.005).any():
            print(f"   (bottle tilt: would touch the scale at {np.degrees(tilt * f):.0f} deg)")
            return None
        if k:
            path.append(qq)
        q = qq
    return path


POUR_MAX_LEAD = 0.015  # grip units (1.4 mm): the command may run at most this far ahead of the jaw
POUR_LOOSEN = 0.004  # grip units (0.4 mm): how far the jaw eases off after the pour, slowly, for the carry home
POUR_OZ_PER_MM = 0.12  # first guess of oz pushed out per mm of squeeze once it flows (run 8: ~0.1-0.15)
POUR_AIM = 0.6  # size each step for this fraction of what is left (the display lags, the dose trails the squeeze)
POUR_MIN_STEP_MM, POUR_MAX_STEP_MM = 0.3, 3.0
POUR_PROBE_FAR, POUR_PROBE_MM, POUR_PROBE_NEAR_MM = 1.5, 0.5, 6.0  # probe steps (mm) before/after 6 mm of squeeze
POUR_MAX_STEPS = 60
POUR_STEP_DRY = 0.0020  # grip units (x 9.5 cm) closed per step before anything flows (~0.19 mm)
POUR_STEP_FLOW = 0.0008  # ... once it flows (~0.08 mm): the flow follows the squeeze
POUR_DT = 0.25  # s between squeeze steps
POUR_LAG = 1.2  # s the display lags the water landing, plus the stream still in the air when the squeeze stops
POUR_RELAX = 0.004  # grip units opened to stop the flow (the bottle re-expands and sucks the spout back)
POUR_SETTLE = 3.0  # s to wait before trusting the reading after a stop
POUR_TOL = 0.1  # oz: done within this of the target (the display's resolution)


def refirm(arm: Arm, q: np.ndarray, target: float, when: str, max_mm: float = 5.0) -> None:
    """Close the held jaw in 0.2 mm steps until the gripper motor's load is back at `target` (at most max_mm)."""
    from pick_bottle import load

    g0, l0 = arm.grip, load(arm, 4)
    while load(arm, 4) < target and (g0 - arm.grip) * JAW_STROKE < max_mm / 1000:
        arm.grip -= 0.002
        arm._cmd(q)
        time.sleep(0.08)
    print(f"   re-firmed {when}: load {l0:.2f} -> {load(arm, 4):.2f}, {(g0 - arm.grip) * JAW_STROKE * 1000:.1f} mm closer")


def spout_tip_px(frame: np.ndarray, roi: np.ndarray | None = None) -> tuple[float, float] | None:
    """Pixel of the wash bottle's spout tip: the orange cap-and-tube blob's point farthest from its thickest part (the
    cap). Only inside `roi` (a mask) when given, and only strongly saturated orange - skin is orange-ish but paler (a hand
    near the dish was taken for the spout). None when no orange blob of a plausible size is seen."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] >= 8) & (hsv[..., 0] <= 25) & (hsv[..., 1] > 160) & (hsv[..., 2] > 110)).astype(np.uint8)
    if roi is not None:
        m &= roi
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[k, cv2.CC_STAT_AREA] < 400:
        return None
    blob = (lab == k).astype(np.uint8)
    dt = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    cy, cx = np.unravel_index(int(np.argmax(dt)), dt.shape)
    ys, xs = np.nonzero(blob)
    j = int(np.argmax((xs - cx) ** 2 + (ys - cy) ** 2))
    return float(xs[j]), float(ys[j])


def aim_spout(arm: Arm, planner: AzPlanner, q: np.ndarray, target: np.ndarray, tip_z: float, tries: int = 2,
              max_fix: float = 0.06) -> np.ndarray | None:
    """With the bottle tilted over the dish: find the spout tip in the detection camera, compare it with `target` (xy)
    at the tip's height, and slide the gripper (orientation kept) to cancel the error. Returns the corrected joints, or
    None when the tip is not found or the error is too large to trust (then do not squeeze)."""
    cam = OPEN_CAMS.get(DET)
    model = FUSED_MODELS.get(DET)
    if cam is None or model is None:
        print("   !! spout aim: the detection camera is not open")
        return None
    out = CAPTURES / "pour"
    out.mkdir(parents=True, exist_ok=True)
    for it in range(tries + 1):
        time.sleep(0.6)
        frame = cam.latest()
        # search only around where the tip should be: 8 cm around the target, from the tip's height down to the rim
        roi = np.zeros(frame.shape[:2], np.uint8)
        th = np.linspace(0, 2 * np.pi, 36)
        for z in (tip_z + 0.06, tip_z, max(0.0, tip_z - 0.06)):
            ring = np.c_[target[0] + 0.08 * np.cos(th), target[1] + 0.08 * np.sin(th), np.full(36, z)]
            cv2.fillConvexPoly(roi, cv2.convexHull(model.project(ring).astype(np.int32)), 1)
        px = spout_tip_px(frame, roi)
        if px is None:
            print("   !! spout aim: no spout tip in view")
            return None
        tip = model.hit_plane(px[0], px[1], tip_z)[:2]
        err = tip - np.asarray(target[:2])
        v = frame.copy()
        cv2.circle(v, (int(px[0]), int(px[1])), 10, (0, 0, 255), 2)
        tp = model.project(np.array([[target[0], target[1], tip_z]]))[0]
        cv2.drawMarker(v, (int(tp[0]), int(tp[1])), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
        cv2.imwrite(str(out / f"aim_{time.strftime('%H%M%S')}_{it}.jpg"), v)
        print(f"   spout aim {it}: tip at ({tip[0]:+.3f}, {tip[1]:+.3f}), {np.linalg.norm(err) * 100:.1f} cm from the target")
        if np.linalg.norm(err) <= 0.008:
            return q
        if np.linalg.norm(err) > max_fix or it == tries:
            if np.linalg.norm(err) > max_fix:
                print(f"   !! spout aim: {np.linalg.norm(err) * 100:.1f} cm off - too far to trust, not pouring")
                return None
            return q
        R = site_rot(planner, q)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = planner.fk_pos(q) - np.r_[err, 0.0]
        q0 = np.zeros(planner.model.nq)
        q0[:6] = q
        good, qq = planner.kin.ik(T, "grasp_site", init_q=q0, limits=planner.limits, max_iters=400,
                                  pos_threshold=1e-3, ori_threshold=2e-2)
        if not good:
            print("   !! spout aim: no IK for the correction")
            return q
        q = np.array(qq[:6])
        arm.glide_path([q], 1.5)
    return q


def loosen(arm: Arm, q: np.ndarray, amount: float, seconds: float = 2.0) -> None:
    """Open the held jaw by `amount` (grip units) in small, slow steps - never a jump the bottle can fall out of."""
    n = max(1, int(round(amount / 0.0005)))
    for _ in range(n):
        arm.grip += amount / n
        arm._cmd(q)
        time.sleep(seconds / n)


def squeeze_pour(arm: Arm, plan: Plan, args: "Args", live: LiveState | None = None) -> float | None:
    """Pour plan.pour["oz"] ounces out of the held squeeze bottle into the bowl on the scale, closed on the scale's
    display (cam2, scale_read.py) with the tactile skin and the gripper motor logged alongside.

    The jaw closes past the held grip in small steps (the pads' skin first, then the bottle's wall); once the
    reading rises, in finer steps. It stops POUR_LAG x the measured flow rate short of the target - what is still
    in the air and what the display has not shown yet - relaxes the squeeze (the bottle re-expands and pulls the
    spout back, so it does not drip), waits for the reading to settle, and tops up in short pulses if it is short.
    Returns the settled reading (oz), or None when the display could not be read."""
    import scale_read
    from pick_bottle import load

    target = float(plan.pour["oz"])
    cam = OPEN_CAMS.get("cam2") or FUSED_CAMS.get("cam2")
    if cam is None:
        print("   !! no cam2 open: cannot read the scale - not pouring")
        return None
    t0 = time.time()
    rows: list[list] = []
    q0 = np.array(arm.last_cmd[:6], dtype=float)

    def reading() -> float | None:
        r = scale_read.read(cam.latest())
        return None if r is None else float(r[0])

    def sample(phase: str, w: float | None) -> None:
        g_pos = float(arm.state7()[6])
        rows.append([round(time.time() - t0, 3), phase, round(arm.grip, 4), round(g_pos, 4), round(load(arm, 2), 3),
                     round(SKIN.magnitude(), 1) if SKIN is not None else "", "" if w is None else w])

    def settled(seconds: float, phase: str) -> float | None:
        vals, end = [], time.time() + seconds
        while time.time() < end:
            w = reading()
            sample(phase, w)
            if w is not None:
                vals.append(w)
            time.sleep(0.12)
        return float(np.median(vals[-5:])) if vals else None

    refirm(arm, q0, args.grip_load, "tilted")  # tilting takes load off the jaw: firm it before it slides
    base = settled(1.5, "tare")
    if base is None:
        print("   !! the scale display cannot be read - not pouring")
        return None
    print(f"   scale before the pour: {base:.1f} oz; pouring {target:.1f} oz -> {base + target:.1f} oz")
    if abs(base) > 1.5:
        # the bowl was tared: a big reading now means the bottle (or the arm) rests on the scale - its weight would
        # swamp the pour and the stop would never come (attempt 1: 10.1 oz). The empty scale's zero itself drifts
        # 0.3-0.8 oz per load cycle (run 7 read 0.6 with nothing touching): the pour counts from `base` anyway
        print(f"   !! the scale reads {base:.1f} oz before pouring (expected ~0 with the bowl tared): something rests "
              "on it - not squeezing")
        return None
    goal = base + target
    q = np.array(arm.last_cmd[:6], dtype=float)
    g_hold = arm.grip  # the held grip: never open wider than this while the bottle hangs in the jaw
    g_min = max(0.0, g_hold - args.pour_max_squeeze / JAW_STROKE)
    s_hold = SKIN.magnitude() if SKIN is not None else None
    s_cap = args.skin_max if args.skin_max is not None else (None if s_hold is None else 3.0 * max(s_hold, 100.0))
    # Dose in steps. The scale's display holds its value while the load changes (run 8: 1.5 oz for 25 s of flow,
    # then 2.2 once it stopped) and a stream's impact spikes it (5.5 oz), so it is only trusted settled. The squeezed
    # bottle is a displacement pump: each step of squeeze pushes out a dose once the air inside is pressurised, and
    # the flow stops by itself once the jaw holds still. Probe in small steps until it flows, then size each step
    # from the measured oz per mm, never re-opening between steps (that lets the collapsed bottle go slack).
    final, k, g_flow, n = base, POUR_OZ_PER_MM, None, 0
    stop_reason = "on target"
    while True:
        poured = final - base
        left = target - poured
        if left <= POUR_TOL / 2:
            break
        n += 1
        if n > POUR_MAX_STEPS:
            stop_reason = f"{POUR_MAX_STEPS} steps"
            break
        sq = (g_hold - arm.grip) * JAW_STROKE * 1000
        if g_flow is None:
            mm = POUR_PROBE_FAR if sq < POUR_PROBE_NEAR_MM else POUR_PROBE_MM
        else:
            mm = float(np.clip(POUR_AIM * left / k, POUR_MIN_STEP_MM, POUR_MAX_STEP_MM))
        g_to = max(g_min, arm.grip - mm / 1000 / JAW_STROKE)
        if g_to >= arm.grip - 1e-6:
            stop_reason = f"the squeeze limit ({args.pour_max_squeeze * 1000:.0f} mm)"
            break
        stalled = 0.0
        while arm.grip > g_to + 1e-6:
            g_pos = float(arm.state7()[6])
            if arm.grip < g_pos - POUR_MAX_LEAD:  # the wall pushes back: wait for the jaw
                stalled += POUR_DT
                if stalled > 3.0:
                    break
            else:
                arm.grip = max(g_to, arm.grip - POUR_STEP_DRY)
                arm._cmd(q)
            sample(f"step{n}", None)
            time.sleep(POUR_DT)
        if stalled > 3.0:
            stop_reason = f"stalled {(g_hold - float(arm.state7()[6])) * JAW_STROKE * 1000:.1f} mm in"
            break
        if live is not None:
            live.set(phase=f"squeeze-pour: step {n}, {poured:.1f} / {target:.1f} oz")
        w = settled(POUR_SETTLE, f"settle{n}")
        if w is None:
            stop_reason = "display unreadable"
            break
        d = w - final
        sq = (g_hold - arm.grip) * JAW_STROKE * 1000
        if g_flow is None and w >= base + 0.2:  # two display counts: one can be a flicker
            g_flow = arm.grip
            print(f"   flowing at {sq:.1f} mm past the held grip" + (f" (skin {SKIN.magnitude():.0f})" if SKIN is not None else ""))
        elif g_flow is not None and d > 0.05:
            k = 0.5 * k + 0.5 * max(0.02, d / mm)  # oz per mm, learned
        final = w
        print(f"   step {n}: +{mm:.1f} mm (squeeze {sq:.1f} mm) -> {w:.1f} oz, poured {w - base:.1f} of {target:.1f}"
              + (f", {k:.2f} oz/mm" if g_flow is not None else ""))
    print(f"   pour done ({stop_reason}): poured {final - base:.1f} of {target:.1f} oz")
    # after the pour: loosen very softly and very little from where the jaw stands. Opening back to the held grip
    # (run 6: 29 mm wider - the squeezed wall does not spring back) leaves the bottle hanging at load 0.06 and it
    # slides out on the way home
    loosen(arm, q, POUR_LOOSEN)
    # the wall creeps and the held load sinks (attempt 2: 0.46 at the grip, 0.10 once tilted, 0.06 after the
    # squeeze - it slid 2 cm down the jaw on the way home): close back up to the hold load, not to a position
    refirm(arm, q, args.grip_load, "after the pour")
    print(f"   after the pour: grip load {load(arm, 2):.2f}" + (f", skin {SKIN.magnitude():.0f}" if SKIN is not None else ""))
    # the log: tactile skin, gripper command / position / load, and the scale, on one clock
    out = CAPTURES / "pour"
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    import csv
    with (out / f"pour_{stamp}.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["t", "phase", "grip_cmd", "grip_pos", "grip_load", "skin_mag", "scale_oz"])
        wr.writerows(rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        R = np.array([[r[0], r[2], r[3], r[4], r[5] if r[5] != "" else np.nan, r[6] if r[6] != "" else np.nan]
                      for r in rows], dtype=float)
        fig, ax = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
        ax[0].plot(R[:, 0], R[:, 5] - base, ".-"); ax[0].axhline(target, ls="--", c="k"); ax[0].set_ylabel("poured (oz)")
        ax[1].plot(R[:, 0], (g_hold - R[:, 1]) * JAW_STROKE * 1000, label="commanded")
        ax[1].plot(R[:, 0], (g_hold - R[:, 2]) * JAW_STROKE * 1000, label="measured"); ax[1].set_ylabel("squeeze (mm)")
        ax[1].legend()
        ax[2].plot(R[:, 0], R[:, 3]); ax[2].set_ylabel("gripper load")
        ax[3].plot(R[:, 0], R[:, 4]); ax[3].set_ylabel("skin |d|"); ax[3].set_xlabel("s")
        fig.suptitle(f"squeeze-pour {target:.1f} oz: settled {'?' if final is None else f'{final - base:.1f}'} oz")
        fig.tight_layout(); fig.savefig(out / f"pour_{stamp}.png", dpi=110); plt.close(fig)
    except Exception as e:
        print(f"   (no pour plot: {e})")
    print(f"   pour log: {out / f'pour_{stamp}.csv'} (+ .png)")
    if live is not None and final is not None:
        live.set(phase=f"poured {final - base:.1f} oz (target {target:.1f})")
    return final


def weigh_readout(seconds: float, live: LiveState | None = None) -> list[Path]:
    """Wait on the scale and photograph its display: cam2 faces it. A frame half-way (the reading settling) and
    one at the end go to captures/weigh_<cam>_<n>.jpg; the display is read from those."""
    shots = []
    # every open camera, fused or not: a camera left out of the fusion (background out of date) still sees
    # the display perfectly well
    views = {k: c for k, c in {**OPEN_CAMS, **FUSED_CAMS}.items() if k in ("cam2", "cam0")}
    for n, dt in enumerate((seconds / 2, seconds / 2)):
        time.sleep(dt)
        for k, c in views.items():
            f = c.latest()
            if f is None:
                continue
            p = CAPTURES / f"weigh_{k}_{n}.jpg"
            cv2.imwrite(str(p), f)
            shots.append(p)
    print(f"   scale display photographed: {', '.join(p.name for p in shots) or 'no camera open'}")
    if live is not None:
        live.set(phase=f"weighed - display photographed ({len(shots)} frames)")
    return shots


def _run_legs(plan: Plan, planner: AzPlanner, arm: Arm, cam: LiveCamera | None, cmodel: CameraModel,
              args: "Args", live: LiveState | None, ws: Workspace, step, i: int) -> bool:
    holding, carry = False, None
    pre_closed = False
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
        if wp.kind == "close" and pre_closed:  # the brute-force search already closed on it
            pre_closed = False
            holding, carry = True, carried_points(plan.obj, plan.z_grasp, grasp_ahead(planner, plan))
            arm.holding = True
            ws.aim(None)
            if args.soft and not secure_hold(arm, planner, plan, args):
                arm.speed = getattr(arm, "speed_free", arm.speed)
                arm.holding = False
                bail(arm, planner, ws, plan, False, "the grasp was lost at the hold test (empty jaw)")
                return False
        elif wp.kind == "close":
            got = False
            if args.soft:
                empty_lift(arm, planner, plan)
            for attempt in range(1, GRASP_TRIES + 1):
                try:  # look before closing: is the object really between the fingers?
                    centre_in_jaw(plan, planner, arm, ws, f"{i}_{attempt}")
                except RuntimeError as e:
                    print(f"   (jaw check skipped: {e})")
                if grip_on_object(arm, args, plan.obj.grip_width):
                    got = True
                    break
                if attempt < GRASP_TRIES and args.refind:
                    # the close may have pushed it: back out, look where it really is now, re-aim - never close
                    # again blind at a spot it may have been knocked away from
                    print(f"   attempt {attempt}/{GRASP_TRIES} closed on nothing: back out and re-measure the object")
                    new = reaim_target(plan, planner, arm, ws, args, i, cam, cmodel)
                    if new is None:
                        arm.holding = False
                        bail(arm, planner, ws, plan, False, "the object could not be re-found and re-aimed safely")
                        return False
                    plan = new
                    continue
                if attempt < GRASP_TRIES:
                    # the camera check fixes left-right, not height: a miss on a low object is the fingers
                    # closing over its rim (the real tips ~1 cm above the model's). Open, step down, retry.
                    here = planner.fk_pos(np.array(arm.last_cmd[:6], dtype=float))
                    z_new = max(GRASP_Z_FLOOR + surface_z(here), here[2] - GRASP_STEP_DOWN)  # never into the scale
                    print(f"   attempt {attempt}/{GRASP_TRIES} closed on nothing: open, step down to "
                          f"{z_new * 100:.1f} cm, look again, retry")
                    arm.set_grip(plan.open_grip, 1.2)
                    if z_new < here[2] - 1e-4:
                        try:
                            move_line(arm, planner, ws, np.r_[here[:2], z_new], plan.yaw, plan.tilt, 1.0)
                        except RuntimeError as e:
                            print(f"   (cannot go lower: {e})")
            if not got:
                arm.holding = False
                bail(arm, planner, ws, plan, False, f"the gripper closed on nothing ({GRASP_TRIES} attempts)")
                return False
            holding, carry = True, carried_points(plan.obj, plan.z_grasp, grasp_ahead(planner, plan))
            arm.holding = True
            ws.aim(None)
            if args.soft and not secure_hold(arm, planner, plan, args):
                arm.speed = getattr(arm, "speed_free", arm.speed)
                arm.holding = False
                bail(arm, planner, ws, plan, False, "the grasp was lost at the hold test (empty jaw)")
                return False
        elif wp.kind == "squeeze":
            if not holding:
                bail(arm, planner, ws, plan, False, "nothing in the gripper to pour from")
                return False
            q_level = np.array(arm.last_cmd[:6], dtype=float)
            P = plan.pour
            tilt_path = None
            if args.pour_tilt_deg > 0:
                # where the bottle is now (level): its axis at the place point, its base pour_base_z up
                base = np.array([plan.place[0], plan.place[1], float(P["base_z"])])
                tip = base + np.array(P["tip"])
                lip = np.array([P["tip_at"][0], P["tip_at"][1], args.pour_lip])
                bowl = tuple(float(v) for v in args.pour_bowl.split(",")) if args.pour_bowl else None
                tilt_path = bottle_tilt_path(planner, ws, q_level, base, tip, lip, np.radians(args.pour_tilt_deg), bowl)
                if tilt_path is None:
                    print("   !! no safe tilt toward the bowl - not pouring")
                else:
                    print(f"   tilting the bottle {args.pour_tilt_deg:.0f} deg toward the bowl, spout tip down to "
                          f"({lip[0]:+.3f}, {lip[1]:+.3f}, {lip[2]:.3f})")
                    if live is not None:
                        live.set(phase=f"tilting {args.pour_tilt_deg:.0f} deg toward the bowl")
                    arm.glide_path(tilt_path, 4.0)
                    time.sleep(0.5)
            aimed = True
            if tilt_path is not None and args.pour_aim:
                # the spout's measured offset is only approximate: find its tip in the detection camera and slide the
                # gripper until it is over the target
                q_aim = aim_spout(arm, planner, tilt_path[-1], np.array(P["tip_at"]), args.pour_lip)
                aimed = q_aim is not None
            if aimed and (args.pour_tilt_deg <= 0 or tilt_path is not None):
                squeeze_pour(arm, plan, args, live)
            if tilt_path is not None:
                arm.glide_path([tilt_path[-1]], 1.5)  # undo any aiming slide
                arm.glide_path(list(reversed(tilt_path[:-1])) + [q_level], 3.0)  # back to level
        elif wp.kind == "hold":
            if args.pour_over and holding:
                do_pour(arm, planner, ws, plan, args)
            if plan.station is not None and wp.label.startswith("weigh"):
                weigh_readout(wp.seconds, live)
            else:
                time.sleep(wp.seconds)
        elif wp.kind == "rest" or (wp.kind == "pose" and not args.blend):
            goto_joint(arm, planner, np.array(REST if wp.kind == "rest" else wp.q), wp.seconds)
        elif args.brute and wp.label == "descend onto the object":
            if not brute_grasp(plan, planner, arm, args, live):
                bail(arm, planner, ws, plan, False, "brute force: no grasp spot held it")
                return False
            pre_closed = True
        elif args.touch_search and wp.label == "descend onto the object":
            if not touch_search(plan, planner, arm):
                bail(arm, planner, ws, plan, False, "the touch search found no spot where the fingers come down beside it")
                return False
        else:
            j = i
            if args.blend and wp.qs and not wp.settle and wp.grip is None:
                # chain the following plain legs into one motion: stop only where something happens
                while j + 1 < len(plan.waypoints):
                    nx = plan.waypoints[j + 1]
                    if nx.kind == "hold" and nx.seconds <= 0 and not (args.pour_over and holding):
                        j += 1
                        continue
                    if nx.kind not in BLEND_KINDS or not nx.qs or nx.label == "descend onto the object":
                        break
                    j += 1
                    if nx.settle or nx.grip is not None:
                        break
                    if nx.kind == "pose" and nx.q is not None and np.allclose(nx.q, REST):
                        break
            if j > i:
                legs = [w for w in plan.waypoints[i:j + 1] if w.kind in BLEND_KINDS]
                qs_all = [q for w in legs for q in w.qs]
                zs = [planner.fk_pos(np.asarray(q, dtype=float))[2] for q in qs_all]
                print(f"   (blended: {' -> '.join(w.label for w in legs)})")
                for k in range(i + 1, j + 1):
                    step(k, plan.waypoints[k].label)
                smooth_follow(arm, planner, ws, qs_all, sum(w.seconds for w in legs), carry,
                              table_floor(min(zs), min(zs)) - 0.004)
                wp = plan.waypoints[j]
                i = j
            elif wp.kind == "pose":
                goto_joint(arm, planner, np.array(wp.q), wp.seconds)
            else:
                move_line(arm, planner, ws, wp.xyz, plan.yaw, plan.tilt, wp.seconds, carry, qs=wp.qs)
            if holding and args.soft and wp.label in ("lift", "lift off the scale"):
                from pick_bottle import load
                e, free = load(arm, 10), getattr(arm, "free_load", 0.0)
                s_now = SKIN.magnitude() if SKIN is not None else None
                s_ref = getattr(arm, "skin_held", None)
                gone_skin = s_now is not None and s_ref is not None and s_ref > 100 and s_now < 0.4 * s_ref
                if e < free + 0.03 or gone_skin:
                    print(f"!! after the lift the jaw reads empty (load {e:.2f}, free {free:.2f}"
                          + ("" if s_now is None else f", skin {s_now:.0f} vs {s_ref:.0f} held") + "): it slid out")
                    arm.holding = False
                    bail(arm, planner, ws, plan, False, "the object slipped out during the lift")
                    return False
                print(f"   still holding after the lift: load {e:.2f}" + ("" if s_now is None else f", skin {s_now:.0f}"))
            if wp.settle:  # a refinement on top of a leg that already arrived: skip it if IK has no answer
                try:
                    pos = settle(arm, planner, wp.xyz[0], wp.xyz[1], wp.xyz[2], plan.yaw, plan.tilt)
                    print(f"   fingertips at {np.round(pos, 3)}")
                except RuntimeError as e:
                    print(f"   (sag compensation skipped: {e})")
            if wp.grip is not None and (wp.release or wp.grip >= GRIP_OPEN - 1e-6) and holding and args.soft:
                from pick_bottle import load
                e, free = load(arm, 10), getattr(arm, "free_load", 0.0)
                tgt = getattr(arm, "hold_load", free + args.grip_load)
                if e < free + 0.04:  # dropped: 0.06-0.07 over a free 0.03-0.05; held and resting on the table: 0.11-0.15
                    print(f"!! at the place point the gripper load is {e:.2f} (free {free:.2f}): the object is not in "
                          "the jaw - it was dropped on the way")
                    arm.speed = getattr(arm, "speed_free", arm.speed)
                    bail(arm, planner, ws, plan, False, "the object was dropped between the pick and the place point")
                    return False
                print(f"   still holding at the place point: load {e:.2f} (free {free:.2f})")
            if wp.grip is not None:
                arm.set_grip(wp.grip, 1.5)
                if (wp.release or wp.grip >= GRIP_OPEN - 1e-6) and holding:
                    arm.speed = getattr(arm, "speed_free", arm.speed)
                    holding, carry = False, None
                    arm.holding = False
                    ws.aim(plan.obj, wp.at or plan.place)
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
    """"x,y,diameter,height" or "x,y,length,width,height,long_side_deg" (m, deg): pick this object instead of a detected one - for a target the survey cannot
    see (clear plastic on the white sheet), measured by hand from the cameras. It becomes object 0; anything
    the survey does detect is kept as an obstacle. More "x,y,d,h" after ";" are extra obstacles (the head of
    a hammer whose handle is the target); detections within 6 cm of any given object are dropped."""
    depth: bool = True
    """First scene analysis with Depth Anything V2 Small (depth.py): metric heights and footprints from the
    detection camera, objects the background survey misses, and a height map the trajectory must clear."""
    side_tilt: tuple[float, ...] = ()
    """Side-grasp wrist tilts to try, in degrees from vertical, in order (e.g. 85 80 75 70 60 for a near-
    horizontal pick of a flat object: the flatter the fingers, the lower they cross it). Default: 60."""
    speed: float = 1.0
    """Scale every arm and jaw motion (2 = twice as fast). The contact-sensing close keeps its own pace."""
    blend: bool = False
    """Travel consecutive plain legs (unfold -> above, lift -> carry -> lower, clear -> fold) as one smooth
    spline instead of stopping at each waypoint; the blended path is collision-checked as travelled."""
    cam_moved_px: float = 3.0
    """How far (px) a camera's fixed background may shift before it counts as moved (~0.7 mm/px at the table)."""
    tip_probe: float = 0.015
    """Extra room (m) assumed beyond each open fingertip when screening grasp poses; the full arm-vs-object
    check at --margin still runs on the whole plan afterwards."""
    table_slack: float = 0.004
    """How far (m) below the commanded fingertip height any part of the arm may reach on a low leg. A slightly
    tilted wrist hangs the finger edges ~6 mm below the fingertip point: 0.007 lets a 1.6 cm tube be taken at
    1.0 cm (lowest part 4 mm above the table)."""
    pour_over: str = ""
    """"x,y": after the lift, pour the held tube over this point (e.g. the Petri dish centre)."""
    pour_open_end: str = ""
    """"x,y" of the tube's open end as it lies on the table (so the pour tips that end down)."""
    pour_lip_z: float = 0.06
    """Height (m) the tube's lip is held at while pouring (the dish top is 1.5 cm)."""
    pour_tilt: float = 100.0
    """Pour tilt, degrees from level (90 = the tube vertical, open end down)."""
    pour_hold: float = 2.5
    """Seconds to hold the full tilt."""
    soft: bool = False
    """Fragile object: close slowly and stop at the first touch (opening lag / motor effort / tactile skin),
    hold with only --soft-preload past it, cap the gripper force at --grip-force, and test the hold by a 5 mm
    lift, tightening 0.5 mm at a time only if it slips."""
    soft_preload: float = 0.005
    grip_load: float = 0.10
    """--soft: gripper motor load (above its free-closing load) to hold the object at. The 9 cm dish flexed out of
    the jaw at ~0.45; 0.10 holds it without bending it."""
    """Gripper units (x 9.5 cm) closed past the first touch in a soft grasp: 0.005 = ~0.5 mm."""
    neck: str | None = None
    """"z_from,diameter" (m): the target has a narrower neck from this height up (a bottle's screw cap) and the
    jaw closes on it - pair with --z-grasp inside the neck. For a squeezable bottle with liquid: the cap is rigid."""
    min_mass: float | None = None
    """--soft: kg the arm must feel hanging in the jaw at the hold test (joint torques, tared by an empty-jaw lift
    at the same pose). Under it the object is sliding in the fingers with its base still down: firm up, retry."""
    skin_max: float | None = None
    """--soft: tactile skin magnitude the hold test must not firm past (default 3x the first-contact reading)."""
    grip_load_max: float | None = None
    """--soft: the most the hold test may firm the grip up to after slips (default --grip-load + 0.10). Keep it low
    for things that deform or spill when squeezed (a filled wash bottle)."""
    grip_force: float | None = None
    """Cap the gripper's blocked force (N; the driver default is 50). --soft sets 10 unless given."""
    brute: bool = False
    """Top grasp: instead of trusting the estimate, try grasp spots on a grid around it (nearest first) until
    one holds; two rounds, re-measuring the object in between. For a Petri dish in a jaw it barely fits."""
    touch_search: bool = False
    """Top grasp: come down to the object in small steps and feel for a finger landing on it (the arm stopping
    short); if so, lift and try further along the jaw. For objects the jaw only just fits (a Petri dish)."""
    side_only: bool = False
    """With --grasp side: never fall back to a top grasp (a horizontal pick was asked for)."""
    max_width: float | None = None
    """Widest object the jaw may close on (default MAX_OBJ_WIDTH, 8.5 cm, a finger margin under the 9.5 cm
    opening); raise it only for one known object that fits (a 9.0 cm dish)."""
    side_cross_max: float = 0.6
    """A side grasp's fingers must cross the object's axis within this fraction of its height (a 1.5 cm dish
    needs ~0.75: the fingertips cannot go lower than ~1 cm)."""
    tactile: bool = False
    """Log the gripper's tactile skin (USB serial, tactile.py) with the gripper motor's position/velocity/
    effort at 100 Hz; saved as meta/tactile/episode_NNNNNN.csv in the dataset, plus a correlation report."""
    tactile_port: str | None = None
    """Serial port of the skin (default: the first /dev/cu.usbmodem*/usbserial* found)."""
    given_only: bool = False
    pick_json: str | None = None
    """A pick file from annotate_pick.py (object bounds + the two fingertip points marked by hand): the target, grasp
    centre, jaw direction, jaw width and grasp height all come from it (a top-down grasp, --given-only)."""
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
    station: bool = False
    """Weigh-station cycle (station.json): pick the object in the taped zone, set it on the scale top, back off
    for --weigh seconds, take it again and put it back where it was. The scale is a fixed box obstacle."""
    refind: str | None = None
    """A finder module (e.g. scripts/bottle_find.py, exposing find(frame, cam, guess_xy)): before every close retry
    the fingers back out, the object is re-measured on the detection camera and the grasp re-aimed at it."""
    refind_offset: str = "0,0"
    """--refind: "dx,dy" m added to a measured position to get the arm's aim (the same arm offset the --given pick
    was corrected by)."""
    refind_max: float = 0.05
    """--refind: the object moved more than this since the grasp was aimed: stop instead of chasing it."""
    fast: bool = False
    """The light planner: same collision sweep but coarser (5 cm IK steps, 4 poses per segment), and no robustness
    re-plans, fused-view check, depth analysis or mid-run rescans. Plans in seconds; use with --given objects."""
    palm: bool = False
    """Seat the object against the palm: full depth only, and the gripper housing may come to 4 mm of the target
    (it is meant to touch it). For big objects (a wash bottle's body) where the whole jaw must be engaged."""
    pour_oz: float | None = None
    """Squeeze-pour this many ounces into a bowl on the scale (read live off its display by cam2, scale_read.py),
    then put the bottle back. Needs --pour-at, --pour-tip."""
    pour_at: str | None = None
    """"x,y": where the spout tip should be (over the bowl, a few cm inside its rim)."""
    pour_tip: str | None = None
    """"dx,dy,dz": the spout tip relative to the bottle's axis at its base (measured before the pick)."""
    pour_base_z: float = 0.07
    """Height of the bottle's base while it pours (above the scale top when the body overhangs the scale)."""
    pour_tilt_deg: float = 35.0
    """Tilt the held bottle this far toward the bowl before squeezing (about the axis square to its spout), so the
    spout points down into it. 0: pour level."""
    pour_lip: float = 0.16
    """Height (m) the spout tip is brought down to at full tilt (a few cm above the bowl's rim)."""
    pour_bowl: str | None = None
    """"x,y,radius,rim_z": the bowl, kept clear of the tilting bottle and arm."""
    pour_aim: bool = True
    """--pour-oz: once tilted, find the spout tip in the detection camera and slide the gripper over the target first
    (no squeeze when the tip is not found or is more than 6 cm off)."""
    pour_load_max: float = 0.85
    """--pour-oz: stop squeezing at this gripper-motor load (1.0 ~ the --grip-force cap)."""
    pour_max_squeeze: float = 0.020
    """m the jaw may close past the held grip while squeezing (a crushing / runaway guard)."""
    station_stage: str = "cycle"
    """--station: "cycle" (onto the scale, weigh, back), "on" (onto the scale, weigh, leave it there), "off" (the
    object already stands on the scale at the place point: take it off and set it down at the --given xy)."""
    station_place: str | None = None
    """--station: "x,y" to set the object down at instead of station.json's place point (must be on the platform) -
    e.g. nearer the base for a heavy object carried high with a flat wrist."""
    weigh: float = 5.0
    """--station: seconds the object sits on the scale with the gripper clear of it."""
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
        ign = station_mask(cmodel, frame.shape[:2])
        objs = survey(frame, cmodel, args.min_area, ignore=ign)
        if args.depth:
            objs = fuse_depth(objs, frame, cmodel, ignore=ign)
    else:  # never overwrite the file we were asked to plan from
        frames = grab_all(args)
        frame = frames[DET]
        CAPTURES.mkdir(exist_ok=True)
        for k, f in frames.items():
            cv2.imwrite(str(CAPTURES / ("survey.jpg" if k == DET else f"survey_{k}.jpg")), f)
        cams = still_cameras(frames, cams)
        frames = {k: f for k, f in frames.items() if k in cams}
        ign = station_ignore(frames, cams)
        objs = survey_multi(frames, cams, args.min_area, ignore=ign)
        if args.depth:
            objs = fuse_depth(objs, frame, cams[DET], ignore=ign[DET] if ign else None,
                              more=grab_more(DET, DEPTH_FRAMES - 1))
    objs = off_station(objs)
    objs = with_given(objs, args)
    if not objs:
        raise RuntimeError(f"nothing in the zone that is not in {BACKGROUND[DET].name} - either it is "
                           "empty, or the background was captured with the objects already standing on it")
    return frame, cmodel, objs


def station_ignore(frames: dict, cams: dict) -> dict | None:
    """Per-camera masks of the weigh station for detection and depth (None without --station)."""
    if STATION is None:
        return None
    return {k: station_mask(cams[k], f.shape[:2], grow=25) for k, f in frames.items() if k in cams}


def off_station(objs: list[Obj]) -> list[Obj]:
    """Drop what was read on or against the weigh station: it is the station."""
    if STATION is None:
        return objs
    fp = np.array(STATION["footprint"])
    # depth smears the scale's top edge into the table beside it: a depth-only blob this close is that
    near = lambda o: poly_sd(o.xy, fp)[0] < (STATION_DEPTH_BAND if o.xy_uncertain else 0.01)
    for o in [o for o in objs if near(o)]:
        print(f"   (ignoring {o.name} at {np.round(o.xy, 3)}: on or against the weigh station)")
    return [o for o in objs if not near(o)]


def with_given(objs: list[Obj], args: "Args") -> list[Obj]:
    """--given prepended as object 0 (with its --neck); a detection within 6 cm of it is the same object and dropped."""
    objs = _with_given(objs, args)
    if args.neck and objs:
        zf, dn = (float(v) for v in args.neck.split(","))
        objs[0].neck = (zf, dn)
        print(f"   {objs[0].name}: gripping its neck - {dn * 100:.1f} cm wide from {zf * 100:.1f} cm up")
    return objs


def _with_given(objs: list[Obj], args: "Args") -> list[Obj]:
    if not args.given:
        return objs
    out = []
    for i, part in enumerate(p for p in args.given.split(";") if p.strip()):
        v = [float(t) for t in part.split(",")]
        name = "given" if i == 0 else f"given_obstacle{i}"
        if len(v) == 6:  # x, y, length, width, height, long-side angle (deg)
            x, y, L, W, H, adeg = v
            L, W = max(L, W), min(L, W)
            out.append(Obj(name, (x, y), L, H, (230, 230, 230), measured=True,
                           length=L, width=W, angle=float(np.radians(adeg)), dims_from="measured by hand"))
        else:
            x, y, d, h = v
            out.append(Obj(name, (x, y), d, h, (230, 230, 230), measured=True, dims_from="measured by hand (round)"))
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
    # the world from the cameras that are still where they were calibrated (a moved cam2 carved a 6 cm
    # bottle into a 24 x 8 cm slab), refined by the depth analysis; every camera is still drawn
    still = still_cameras(frames, cams)
    ign = station_ignore(frames, still)
    objs = survey_multi({k: f for k, f in frames.items() if k in still}, still, args.min_area, ignore=ign)
    if args.depth and DET in still:
        objs = fuse_depth(objs, frames[DET], still[DET], ignore=ign[DET] if ign else None,
                          more=grab_more(DET, DEPTH_FRAMES - 1))
    objs = off_station(objs)
    for o in objs:
        print(f"   {o.describe()}")
    meshes = []  # live_map.build poses the arm itself; scene.json only needs the q
    scene = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frame": "robot base: x forward, y left, z up, table z=0 (m)",
        "cameras": [camera_entry(k, cams[k], frames[k], raw[k]) for k in sorted(cams)],
        "objects": [{"name": o.name, "axis": list(o.xy), "diameter": o.diameter, "height": o.height,
                     "color": list(o.color), "dims": o.dims_from} for o in objs],
        "robot": {"q": [0.0] * 6, "tcp": [0.0, 0.0, 0.0], "meshes": meshes},
        "calibration_points": [r["fk"] for d in ("sweep_zone2", "sweep_zone3", "sweep_zone6")
                               if (CAPTURES / d / "sweep.json").exists()
                               for r in json.loads((CAPTURES / d / "sweep.json").read_text())],
    }
    zone = load_zone()
    if zone is not None:
        from fused import TopView, jpeg_b64
        tv = TopView(still, zone)
        top = tv.annotate(tv.render(frames), zone, scene["objects"])
        cv2.imwrite(str(CAPTURES / "fused_top.jpg"), top)
        scene["top"] = {"b64": jpeg_b64(top), "meta": tv.meta()}
    (HERE / "scene.json").write_text(json.dumps(scene))
    print(f"wrote scene.json: {len(scene['cameras'])} cameras, {len(objs)} object(s)")
    try:  # the sensors panel and the depth world (camera and serial reads only)
        import map_world
        map_world.build()
    except Exception as e:
        print(f"   (sensors/world report skipped: {e})")
    live_map.build(HERE / "scene.json")


def stage_background(args: Args) -> None:
    """Capture the empty-table reference both this script and scene3d.py subtract. Everything that is on
    the table now becomes invisible to detection, so the table must be clear."""
    CAPTURES.mkdir(exist_ok=True)
    for key, idx in (("cam0", 0), ("cam1", 1), ("cam2", 2), ("cam3", 3)):  # camN (YAM_CAM_MAP picks the device)
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


MIN_TARGET_H = 0.025  # m: lower than this is a cable end, a sheet or a shadow, not the thing to pick


def choose_target(objs: list[Obj], args: Args) -> int:
    """--object 0 with several objects and nothing hand-measured: the thing the person put there, i.e. the
    largest object the jaw can close on - not whatever the survey listed first (a cable end hanging into the
    zone, 3 cm tall, once took the place of a 19 cm bottle). The rest stay obstacles."""
    if args.object != 0 or args.given or len(objs) < 2:
        return args.object
    ok = [i for i, o in enumerate(objs) if o.height >= MIN_TARGET_H and o.grip_width <= MAX_OBJ_WIDTH]
    if not ok:
        return 0
    i = max(ok, key=lambda k: objs[k].height * objs[k].diameter ** 2)
    if i != 0:
        print(f"   target: {objs[i].name} (the largest graspable object; "
              f"{', '.join(o.name for k, o in enumerate(objs) if k != i)} kept as obstacles)")
    return i


def plan_from_frame(frame: np.ndarray, cmodel: CameraModel, objs: list[Obj], args: Args) -> Plan:
    if not 0 <= args.object < len(objs):
        raise RuntimeError(f"--object {args.object} out of range: {len(objs)} object(s) found")
    print(f"{len(objs)} object(s) on the table")
    plan = make_plan(objs, choose_target(objs, args), args, AzPlanner())
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
            for i in range(4):
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
        arm.speed = float(args.speed)
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
            if args.depth:
                more = []
                for _ in range(DEPTH_FRAMES - 1):
                    time.sleep(0.35)
                    more.append(cams[DET].latest())
                objs = fuse_depth(objs, frames[DET], models[DET], ignore=masks.get(DET), more=more)
            FUSED_CAMS.update({k: cams[k] for k in models if k in cams})
            OPEN_CAMS.update(cams)
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
    force = args.grip_force if args.grip_force is not None else (10.0 if args.soft else None)
    if force is not None:
        from pick_bottle import set_gripper_force
        print(f"gripper force capped at {force:.0f} N" if set_gripper_force(arm, force)
              else "(could not cap the gripper force: no limiter in this driver)")
    grip_log = skin = None
    if args.tactile:  # the gripper's skin next to its motor, on one clock (tactile.py)
        from tactile import GripLog, TactileSkin
        try:
            skin = TactileSkin(args.tactile_port)
            skin.tare()  # the jaw is open and touching nothing yet
            grip_log = GripLog(skin, arm.robot.get_observations, lambda: float(arm.last_cmd[6]),
                               lambda: str(live.doc.get("phase", "")))
            grip_log.start()
            global SKIN
            SKIN = skin
            print(f"tactile skin on {skin.port}: {skin.width} channels, logging with the gripper motor")
        except Exception as e:  # the pick does not depend on the skin
            print(f"!! tactile skin not logged: {e}")
            skin = grip_log = None
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
        if grip_log:
            grip_log.stop()
            where = (Path(args.record) / "meta" / "tactile" / f"episode_{rec.ep:06d}.csv" if rec
                     else CAPTURES / "tactile" / f"{live.doc.get('run', 'run')}.csv")
            try:
                print(f"saved skin + gripper motor log: {grip_log.save(where)} ({len(grip_log.rows)} samples)")
                from tactile import report
                report(where, where.with_suffix(".png"))
            except Exception as e:
                print(f"!! could not save the tactile log: {e}")
            skin.close()
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
    global JAW_ACROSS, MAX_OBJ_WIDTH, SIDE_TILTS, TABLE_SLACK, CAM_MOVED_PX, TIP_PROBE, STATION
    global TARGET_TOL, TALL_TOL, SEAT_BACKOFFS, FAST, VALIDATE_STEP, SCAN_N
    TIP_PROBE = float(args.tip_probe)
    if args.pick_json:
        pk = json.loads(Path(args.pick_json).read_text())
        g, o = pk["grasp"], pk["object"]
        target = f"{g['centre'][0]:.4f},{g['centre'][1]:.4f},{max(g['width'], 0.008):.4f},{max(o['height'], g['z'] + 0.005):.4f}"
        # anything else already given stays an obstacle after the marked target
        args.given = target + (";" + args.given if args.given else "")
        args.given_only, args.object = True, 0
        args.grasp, args.z_grasp = "top", float(g["z"])
        args.jaw_across = f"{g['jaw_across'][0]:.4f},{g['jaw_across'][1]:.4f}"
        print(f"pick file {args.pick_json}: grasp at {g['centre']} z {g['z'] * 1000:.0f} mm, jaw {g['width'] * 1000:.0f} mm "
              f"closing along {g['close_dir']}; object {o['height'] * 1000:.0f} mm tall")
    if args.fast:
        FAST, VALIDATE_STEP, SCAN_N = True, 0.05, 4
        args.depth, args.rescan = False, False
        print("fast planner: coarse sweep, no robustness re-plans / fused-view check / depth / rescans")
    if args.palm:  # the palm is meant to touch the object: seat it fully, no shallower fallbacks
        TARGET_TOL = TALL_TOL = 0.004
        SEAT_BACKOFFS = (0.0,)
        print("palm grasp: the object is seated against the gripper housing (full finger depth)")
    if args.pour_oz is not None:
        global POUR
        tx, ty = (float(v) for v in args.pour_at.split(","))
        dx, dy, dz = (float(v) for v in args.pour_tip.split(","))
        POUR = {"oz": float(args.pour_oz), "base_z": float(args.pour_base_z), "tip": [dx, dy, dz],
                "tip_at": [tx, ty], "axis": [tx - dx, ty - dy]}
        print(f"squeeze-pour {args.pour_oz:.1f} oz: spout tip at ({tx:+.3f}, {ty:+.3f}), "
              f"{(args.pour_base_z + dz) * 100:.0f} cm up; the bottle axis at ({tx - dx:+.3f}, {ty - dy:+.3f})")
    if args.station:
        STATION = load_station()
        if args.station_place:
            xy = [float(v) for v in args.station_place.split(",")]
            if poly_sd(np.array(xy), np.array(STATION["footprint"]))[0] > -0.05:
                raise SystemExit(f"--station-place {xy} is not at least 5 cm inside the scale platform")
            STATION["place"] = xy
        if args.task == Args.task:
            args.task = "pick up the object, set it on the scale, then put it back where it was"
        print(f"weigh station: top {STATION['top_z'] * 100:.1f} cm, place point {STATION['place']}, "
              f"{args.weigh:.0f} s on the scale")
    CAM_MOVED_PX = float(args.cam_moved_px)
    TABLE_SLACK = float(args.table_slack)
    SIDE_TILTS = tuple(float(np.radians(t)) for t in args.side_tilt)
    if args.max_width is not None:
        MAX_OBJ_WIDTH = min(float(args.max_width), JAW_STROKE - 0.003)
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
