"""The camera-fused workspace: one bird's-eye image of the zone built from every calibrated camera.

    python fused.py              # grab every calibrated camera -> captures/fused_top.jpg

Each camera sees the table plane (z = 0) through a homography plus its lens distortion. For every pixel of
the top view the table point is projected into every camera (distortion included, so this is exact, not
a 4-point warp), and the camera whose view there still matches its own empty-zone background wins, ties
going to the steeper view. So the arm or an object leaning into one camera's perspective is not painted
onto the table where another camera can see past it. Only the table plane is true in this image; where a
single camera covers the table a tall object still leans, so objects are drawn from the carved 3D model.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np

from calib3d import CameraModel

HERE = Path(__file__).resolve().parent
PX_PER_M = 1500.0  # top-view resolution: 1.5 px per mm
MARGIN = 0.04  # m of table around the zone


def bounds(zone: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return zone.min(axis=0) - MARGIN, zone.max(axis=0) + MARGIN


class TopView:
    """Precomputed per-camera remap tables from the top-view grid into each camera image."""

    def __init__(self, cams: dict[str, CameraModel], zone: np.ndarray, px_per_m: float = PX_PER_M):
        self.lo, self.hi = bounds(zone)
        self.s = px_per_m
        # image rows run along -x (far side of the zone at the top, the robot base at the bottom would be
        # +x... keep it intuitive for someone standing at the camera side: +x (away from the base) at the
        # bottom, +y (robot's left) to the right) -> row = (x - lo.x), col = (hi.y - y)
        self.h = int(np.ceil((self.hi[0] - self.lo[0]) * px_per_m))
        self.w = int(np.ceil((self.hi[1] - self.lo[1]) * px_per_m))
        rows, cols = np.mgrid[0 : self.h, 0 : self.w]
        x = self.lo[0] + (rows + 0.5) / px_per_m
        y = self.hi[1] - (cols + 0.5) / px_per_m
        P = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
        self.maps: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for k, cam in cams.items():
            W, H = cam.size
            front = (P @ cam.R.T + cam.t)[:, 2] > 0.05
            uv = np.full((len(P), 2), -1.0, np.float32)
            uv[front] = cam.project(P[front])
            ok = front & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
            ray = P - cam.C  # weight: how steeply the camera looks down at that point
            wgt = np.where(ok, np.abs(ray[:, 2]) / np.linalg.norm(ray, axis=1), 0.0) ** 2
            self.maps[k] = (uv[:, 0].reshape(self.h, self.w), uv[:, 1].reshape(self.h, self.w),
                            wgt.reshape(self.h, self.w).astype(np.float32))
        # each camera's empty-zone background, seen through the same table-plane map
        from scene3d import BACKGROUND
        self.bgs: dict[str, np.ndarray] = {}
        for k, (mx, my, _) in self.maps.items():
            bg = cv2.imread(str(BACKGROUND[k])) if k in BACKGROUND else None
            if bg is not None:
                self.bgs[k] = cv2.remap(bg, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def to_px(self, xy) -> tuple[int, int]:
        """Robot-frame (x, y) -> top-view pixel (col, row)."""
        return int(round((self.hi[1] - xy[1]) * self.s)), int(round((xy[0] - self.lo[0]) * self.s))

    def render(self, frames: dict[str, np.ndarray]) -> np.ndarray:
        """Composite by agreement with the empty table, not by averaging: at each point take the camera
        whose view there still matches its own empty-zone background. Something standing in front of one
        camera (the arm, a tall object leaning in the perspective) is then not painted onto the table; where
        every camera sees a change - a real footprint - the steepest camera's view is kept."""
        best = np.full((self.h, self.w), np.inf, np.float32)
        out = np.full((self.h, self.w, 3), 24, np.uint8)
        for k, (mx, my, wgt) in self.maps.items():
            if k not in frames or frames[k] is None:
                continue
            img = cv2.remap(frames[k], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            bg = self.bgs.get(k)
            change = (cv2.absdiff(img, bg).max(axis=2).astype(np.float32) if bg is not None
                      else np.zeros(best.shape, np.float32))
            score = np.where(wgt > 0, change - 40.0 * wgt, np.inf)  # ties go to the steeper view
            pick = score < best
            out[pick] = img[pick]
            best[pick] = score[pick]
        return out

    def annotate(self, img: np.ndarray, zone: np.ndarray, objects: list[dict]) -> np.ndarray:
        """Zone outline and the carved objects on top of the table-plane image."""
        out = img.copy()
        cv2.polylines(out, [np.array([self.to_px(p) for p in zone], np.int32)], True, (215, 111, 47), 2)
        for o in objects:
            c = self.to_px(o["axis"])
            r = max(2, int(o["diameter"] / 2 * self.s))
            cv2.circle(out, c, r, (60, 60, 230) if o.get("role") == "target" else (180, 60, 190), 2)
            cv2.putText(out, f"{o['name']} {o['height'] * 100:.0f}cm", (c[0] + r + 3, c[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 3)
            cv2.putText(out, f"{o['name']} {o['height'] * 100:.0f}cm", (c[0] + r + 3, c[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return out

    def meta(self) -> dict:
        """Where the image sits in the robot frame, for the map: corners (x, y) of the image's
        top-left, top-right, bottom-right, bottom-left."""
        lo, hi = self.lo, self.hi
        return {"corners": [[float(lo[0]), float(hi[1])], [float(lo[0]), float(lo[1])],
                            [float(hi[0]), float(lo[1])], [float(hi[0]), float(hi[1])]],
                "px_per_m": self.s, "size": [self.w, self.h]}


def jpeg_b64(img: np.ndarray, quality: int = 70) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


if __name__ == "__main__":
    from calib3d import load_cameras
    from scene3d import grab

    cams = load_cameras()
    zone = np.array(json.loads((HERE / "zone.json").read_text())["polygon"])
    tv = TopView(cams, zone)
    frames = {k: grab(int(k[3:])) for k in cams}
    img = tv.render(frames)
    cv2.imwrite(str(HERE / "captures" / "fused_top.jpg"), tv.annotate(img, zone, []))
    print(f"wrote captures/fused_top.jpg ({tv.w}x{tv.h} px, {len(frames)} cameras)")


# ------------------------------------------------------------------ top / side / isometric views of a plan
def _iso(p: np.ndarray) -> np.ndarray:
    """Isometric projection of robot-frame points (N, 3) onto 2D (right, up), looking from the front-right
    (+x, -y) and above, like the cameras."""
    a = np.radians(30)
    right = (p[:, 0] * np.sin(np.radians(45)) + p[:, 1] * np.cos(np.radians(45)))
    up = p[:, 2] - (p[:, 0] * np.cos(np.radians(45)) - p[:, 1] * np.sin(np.radians(45))) * np.sin(a)
    return np.c_[right, up]


def render_views(top_img: np.ndarray, tv: "TopView", zone: np.ndarray, objects: list[dict], path: np.ndarray,
                 swept: np.ndarray, bad: np.ndarray, title: str) -> np.ndarray:
    """One image with three panels of the same fused scene: TOP (the multi-camera homography of the table),
    SIDE (x-z) and ISOMETRIC. objects: [{"axis", "diameter", "height", "role"}]; path: planned fingertip
    polyline (N, 3); swept: every arm point along the joint path (M, 3); bad: bool (M,) - inside a clearance."""
    H = 520
    # --- top: the camera image of the table plane, swept arm footprint, path, objects
    top = tv.annotate(top_img, zone, objects)
    over = top.copy()
    for pts, col in ((swept[~bad], (170, 170, 170)), (swept[bad], (0, 0, 255))):
        for x, y, _ in pts[:: max(1, len(pts) // 20000)]:
            c = tv.to_px((x, y))
            if 0 <= c[0] < top.shape[1] and 0 <= c[1] < top.shape[0]:
                cv2.circle(over, c, 1, col, -1)
    top = cv2.addWeighted(over, 0.45, top, 0.55, 0)
    if len(path) > 1:
        cv2.polylines(top, [np.array([tv.to_px(p) for p in path], np.int32)], False, (0, 170, 255), 2)
    top = cv2.resize(top, (int(top.shape[1] * H / top.shape[0]), H))

    def canvas(proj, w=560):
        allp = np.vstack([proj(swept)] + [proj(path)] if len(path) else [proj(swept)])
        lo, hi = allp.min(axis=0) - 0.03, allp.max(axis=0) + 0.03
        s = min((w - 20) / (hi[0] - lo[0]), (H - 40) / (hi[1] - lo[1]))
        img = np.full((H, w, 3), 245, np.uint8)
        to = lambda q: (int(10 + (q[0] - lo[0]) * s), int(H - 10 - (q[1] - lo[1]) * s))
        return img, to

    # --- side (x-z): looking from -y
    side, to = canvas(lambda p: np.c_[p[:, 0], p[:, 2]])
    cv2.line(side, to((-0.2, 0)), to((0.8, 0)), (120, 100, 80), 2)
    zx = [zone[:, 0].min(), zone[:, 0].max()]
    cv2.line(side, to((zx[0], 0)), to((zx[1], 0)), (215, 111, 47), 5)
    for pts, col in ((swept[~bad], (175, 175, 175)), (swept[bad], (0, 0, 255))):
        for x, _, z in pts[:: max(1, len(pts) // 20000)]:
            cv2.circle(side, to((x, z)), 1, col, -1)
    for o in objects:
        r, h, cx = o["diameter"] / 2, o["height"], o["axis"][0]
        cv2.rectangle(side, to((cx - r, h)), to((cx + r, 0)), (60, 60, 230) if o.get("role") == "target" else (180, 60, 190), 2)
    if len(path) > 1:
        cv2.polylines(side, [np.array([to((p[0], p[2])) for p in path], np.int32)], False, (0, 170, 255), 2)
    # --- isometric
    iso, to = canvas(_iso)
    zp = np.c_[zone, np.zeros(len(zone))]
    cv2.polylines(iso, [np.array([to(q) for q in _iso(zp)], np.int32)], True, (215, 111, 47), 2)
    for pts, col in ((swept[~bad], (175, 175, 175)), (swept[bad], (0, 0, 255))):
        for q in _iso(pts[:: max(1, len(pts) // 20000)]):
            cv2.circle(iso, to(q), 1, col, -1)
    th = np.linspace(0, 2 * np.pi, 36)
    for o in objects:
        r, h = o["diameter"] / 2, o["height"]
        col = (60, 60, 230) if o.get("role") == "target" else (180, 60, 190)
        for z in (0.0, h):
            ring = np.c_[o["axis"][0] + r * np.cos(th), o["axis"][1] + r * np.sin(th), np.full(len(th), z)]
            cv2.polylines(iso, [np.array([to(q) for q in _iso(ring)], np.int32)], True, col, 2)
    if len(path) > 1:
        cv2.polylines(iso, [np.array([to(q) for q in _iso(np.asarray(path))], np.int32)], False, (0, 170, 255), 2)
    out = np.hstack([top, side, iso])
    for x0, name in ((0, "TOP - fused homography"), (top.shape[1], "SIDE - x/z"), (top.shape[1] + 560, "ISOMETRIC")):
        cv2.rectangle(out, (x0, 0), (x0 + 260, 26), (255, 255, 255), -1)
        cv2.putText(out, name, (x0 + 6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
    band = np.full((34, out.shape[1], 3), 255, np.uint8)
    cv2.putText(band, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200) if bad.any() else (30, 120, 30), 2)
    return np.vstack([band, out])
