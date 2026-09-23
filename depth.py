"""Metric depth for the first scene analysis: Depth Anything V2 Small, pinned to the calibrated table.

    python depth.py                 # grab cam1, print every object's L x W x H, write captures/depth_*.jpg
    python depth.py --frame F.jpg   # the same on a saved frame

The weights are not in git (99 MB). Fetch them once:
    curl -L -o models/depth_anything_v2_small.onnx \
      https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx
(sha256 afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c) and `uv pip install onnxruntime`.

Depth Anything V2 (depth-anything-v2.github.io; the Small model, Apache-2.0, run with ONNX Runtime from
models/depth_anything_v2_small.onnx) predicts *relative* inverse depth: right up to an unknown scale and shift.
The zone's table plane is known exactly in every calibrated camera, so the empty table pixels give the true
inverse depth there, and a robust fit of  pred = a / Z + b  on them turns the whole image metric. Every zone
pixel then becomes a 3D point in the robot frame; binned into a 5 mm height map, whatever rises above the
sheet is an object - white and colourless ones included, which background subtraction loses on the white
sheet - and its 3D points give its footprint (length x width, long-side direction) and height from one view.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE / "models" / "depth_anything_v2_small.onnx"
PREPROC = HERE / "models" / "depth_anything_v2_small.preprocessor.json"
CELL = 0.005  # m: height-map resolution
MIN_RISE = 0.012  # m above the sheet: below this is paper, tape and depth noise
MIN_CELLS = 12  # 3 cm^2: smaller blobs are noise
FLY_STEP = 0.08  # ln-depth range across a FLY_WIN window above which a pixel sits on a depth step
FLY_WIN = 7


class DepthModel:
    def __init__(self, path: Path = MODEL):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        cfg = json.loads(PREPROC.read_text())
        self.size = int(cfg["size"]["height"])
        self.mean = np.array(cfg["image_mean"], np.float32)
        self.std = np.array(cfg["image_std"], np.float32)

    def _shape(self, h: int, w: int) -> tuple[int, int]:
        # DPTImageProcessor(keep_aspect_ratio, ensure_multiple_of=14): the scale closer to 1 wins
        sh, sw = self.size / h, self.size / w
        s = sh if abs(1 - sh) < abs(1 - sw) else sw
        return max(14, int(round(h * s / 14)) * 14), max(14, int(round(w * s / 14)) * 14)

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        """Relative inverse depth (larger = nearer), at the frame's own resolution."""
        h, w = bgr.shape[:2]
        th, tw = self._shape(h, w)
        x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (tw, th), interpolation=cv2.INTER_CUBIC)
        x = ((x.astype(np.float32) / 255.0 - self.mean) / self.std).transpose(2, 0, 1)[None]
        out = self.sess.run(None, {self.inp: x})[0].squeeze()
        return cv2.resize(out.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


_MODEL: DepthModel | None = None


def model() -> DepthModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = DepthModel()
    return _MODEL


def _rays(cam, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Undistorted normalised camera rays (x, y, 1) for pixels (u, v)."""
    n = cv2.undistortPoints(np.c_[u, v].astype(np.float64).reshape(-1, 1, 2), cam.K, cam.dist).reshape(-1, 2)
    return np.c_[n, np.ones(len(n))]


def table_depth(cam, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Camera-frame Z of the table plane (robot z = 0) seen through pixels (u, v)."""
    r = _rays(cam, u, v)  # camera frame
    d = r @ cam.R  # robot-frame directions (R^T r), not normalised: camera Z scales with it
    s = -cam.C[2] / d[:, 2]  # C + s d hits z = 0
    return s  # the camera-frame ray (x, y, 1) scaled by s: Z = s


def metric(cam, frame: np.ndarray, rel: np.ndarray, table_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Camera Z per pixel from relative inverse depth, fitted on `table_mask` (empty table pixels)."""
    v, u = np.nonzero(table_mask)
    if len(u) > 40000:
        k = np.random.default_rng(0).choice(len(u), 40000, replace=False)
        u, v = u[k], v[k]
    z = table_depth(cam, u, v)
    ok = z > 0
    inv, p = 1.0 / z[ok], rel[v[ok], u[ok]]
    keep = np.ones(len(p), bool)
    for _ in range(4):  # least squares, then drop the worst residuals (edges, shadows, stray objects)
        A = np.c_[inv[keep], np.ones(keep.sum())]
        (a, b), *_ = np.linalg.lstsq(A, p[keep], rcond=None)
        res = np.abs(a * inv + b - p)
        keep = res <= np.percentile(res, 80)
    fit = {"a": float(a), "b": float(b), "pixels": int(keep.sum()),
           "table_rms_mm": float(np.sqrt(np.mean(((1.0 / np.maximum((p[keep] - b) / a, 1e-6)) - 1.0 / inv[keep]) ** 2)) * 1000)}
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = 1.0 / ((rel - b) / a)
    Z[~np.isfinite(Z) | (Z <= 0)] = np.nan
    return Z, fit


def points(cam, Z: np.ndarray, mask: np.ndarray, with_px: bool = False):
    """Robot-frame 3D points of the pixels in `mask` (and their pixel coordinates with with_px)."""
    v, u = np.nonzero(mask & np.isfinite(Z))
    P_cam = _rays(cam, u, v) * Z[v, u][:, None]
    P = (P_cam - cam.t.reshape(1, 3)) @ cam.R  # R^T (P - t)
    return (P, np.c_[u, v]) if with_px else P


def heightmap(P: np.ndarray, zone: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell height (90th percentile of the points in it) over the zone's bounding box."""
    lo, hi = zone.min(axis=0) - 0.02, zone.max(axis=0) + 0.02
    nx, ny = int(np.ceil((hi[0] - lo[0]) / CELL)), int(np.ceil((hi[1] - lo[1]) / CELL))
    ix = ((P[:, 0] - lo[0]) / CELL).astype(int)
    iy = ((P[:, 1] - lo[1]) / CELL).astype(int)
    m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    H = np.full((nx, ny), np.nan)
    key = ix[m] * ny + iy[m]
    order = np.argsort(key)
    key, z = key[order], P[m, 2][order]
    starts = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
    for s, e in zip(starts, np.r_[starts[1:], len(key)]):
        if e - s >= 3:
            H.flat[key[s]] = np.percentile(z[s:e], 90)
    return H, lo, np.array([nx, ny])


def _rect(xy: np.ndarray):
    """Smallest rectangle around cell centres: centre (m), length >= width (m, cell edges), long-side angle."""
    (cx, cy), (a, b), ang = cv2.minAreaRect((xy * 1000).astype(np.float32))
    a, b = a / 1000 + CELL, b / 1000 + CELL
    th = np.radians(ang)
    if a < b:
        a, b, th = b, a, th + np.pi / 2
    return (cx / 1000, cy / 1000), a, b, th


def objects(H: np.ndarray, lo: np.ndarray, zone: np.ndarray) -> list[dict]:
    """Everything that rises MIN_RISE above the sheet: centre, L x W (m), long-side angle (rad), height."""
    from pick_place import zone_margin

    up = np.nan_to_num(H, nan=0.0) > MIN_RISE
    up = cv2.morphologyEx(up.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(up, connectivity=8)
    out = []
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] < MIN_CELLS:
            continue
        ij = np.argwhere(lab == c)
        xy = lo + (ij + 0.5) * CELL
        if zone_margin(xy.mean(axis=0), zone) < -0.01:
            continue
        hs = H[lab == c]
        top_h = float(np.nanpercentile(hs, 95))
        # One view sees an object's front and top, not its back: along the line of sight a round object
        # loses its hidden half and a tall box gains a trail of blended depth. A flat top (a box, a vial) is
        # the footprint itself, seen whole, so when the top face covers much of the blob, its rectangle is
        # the dimensions. Otherwise (a bottle's cap, a handle's ridge) the whole blob's rectangle is used and
        # a near-square one is taken as round, its size the larger side.
        high = hs >= 0.8 * top_h
        rect_all = _rect(xy)
        shape = "whole blob"
        if high.sum() >= 6 and high.sum() >= 0.35 * len(hs):
            (cx, cy), a, b, th = _rect(xy[high])
            shape = "flat top"
        else:
            (cx, cy), a, b, th = rect_all
            if a < 1.6 * b:
                b = a
                shape = "round"
        out.append({"xy": (float(cx), float(cy)), "length": float(a), "width": float(b),
                    "angle": float((th + np.pi / 2) % np.pi - np.pi / 2), "height": top_h,
                    "cells": int(len(ij)), "shape": shape})
    return out


def _hidden_hull(o: dict, C: np.ndarray) -> np.ndarray:
    """The patch of table an object hides from a camera at C: its footprint plus its top corners pushed along
    the camera's rays down to the table (a convex hull, robot-frame xy)."""
    c, s_ = np.cos(o["angle"]), np.sin(o["angle"])
    ax, ay = np.array([c, s_]), np.array([-s_, c])
    corners = [np.array(o["xy"]) + a * o["length"] / 2 * ax + b * o["width"] / 2 * ay for a in (-1, 1) for b in (-1, 1)]
    pts = []
    for q in corners:
        pts.append(q)
        top = np.array([q[0], q[1], o["height"]])
        t = C[2] / max(C[2] - o["height"], 1e-3)  # C + t (top - C) reaches z = 0
        pts.append((C + t * (top - C))[:2])
    return cv2.convexHull(np.array(pts, np.float32))


def drop_hidden(objs: list[dict], H: np.ndarray, lo: np.ndarray, C: np.ndarray,
                ratio: float = 2.0) -> tuple[list[dict], np.ndarray]:
    """What the camera cannot see is unknown, not occupied. Behind a tall object the network extrapolates
    the hidden table, and it came out 2 cm high: a phantom 'object' 11 cm behind a wash bottle that blocked
    every approach to the bottle. Objects less than 1/ratio as tall as one that hides them, and every
    height-map cell hidden behind a taller object (outside its own footprint), are dropped."""
    H = H.copy()
    keep = []
    hulls = [(o, _hidden_hull(o, C)) for o in objs]
    for o in objs:
        hidden = any(t is not o and t["height"] >= ratio * o["height"] and
                     cv2.pointPolygonTest(h, (float(o["xy"][0]), float(o["xy"][1])), False) >= 0
                     for t, h in hulls)
        if not hidden:
            keep.append(o)
    for t, h in hulls:  # the hidden table behind each kept object: no data, so no height
        if t not in keep:
            continue
        own = cv2.convexHull(np.array([np.array(t["xy"]) + a * t["length"] / 2 * np.array([np.cos(t["angle"]), np.sin(t["angle"])])
                                        + b * t["width"] / 2 * np.array([-np.sin(t["angle"]), np.cos(t["angle"])])
                                        for a in (-1.2, 1.2) for b in (-1.2, 1.2)], np.float32))
        for i, j in np.argwhere(np.isfinite(H)):
            xy = (float(lo[0] + (i + 0.5) * CELL), float(lo[1] + (j + 0.5) * CELL))
            if cv2.pointPolygonTest(h, xy, False) >= 0 and cv2.pointPolygonTest(own, xy, False) < 0 and H[i, j] < 0.8 * t["height"]:
                H[i, j] = np.nan
    return keep, H


def analyse(frame: np.ndarray, cam, key: str = "cam1", ignore: np.ndarray | None = None,
            save: bool = True) -> dict:
    """The first scene analysis: metric depth for one calibrated view -> objects with dimensions."""
    # numpy on Apple's Accelerate BLAS raises spurious divide/overflow warnings from ordinary matmuls
    with np.errstate(all="ignore"):
        return _analyse(frame, cam, key, ignore, save)


def _analyse(frame, cam, key, ignore, save) -> dict:
    import pick_place as pp

    zone = pp.load_zone()
    t = time.time()
    rel = model()(frame)
    t_inf = time.time() - t
    win = pp.workspace_mask(cam, frame.shape[:2]) > 0
    # the sheet itself: the zone polygon on the table, minus anything the background says has changed
    poly = cam.project(np.c_[zone, np.zeros(len(zone))]).astype(np.int32)
    sheet = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(sheet, [poly], 255)
    bg = cv2.imread(str(pp.BACKGROUND[key]))
    changed = cv2.dilate(pp.foreground(frame, bg, None), np.ones((25, 25), np.uint8)) if bg is not None else 0
    table = (sheet > 0) & (changed == 0)
    if ignore is not None:
        table &= ignore == 0
        win &= ignore == 0
    Z, fit = metric(cam, frame, rel, table)
    # "flying pixels": where a tall object's top meets the far table, the network blends the two depths, and
    # those pixels land in the air behind the object - a trail that doubled the mouse box's footprint. Drop
    # every pixel on a sharp relative depth step.
    lz = np.log(np.where(np.isfinite(Z), Z, np.nanmedian(Z))).astype(np.float32)
    k = np.ones((FLY_WIN, FLY_WIN), np.uint8)
    edge = (cv2.dilate(lz, k) - cv2.erode(lz, k)) > FLY_STEP  # depth range across a small window
    P, px = points(cam, Z, win & ~edge, with_px=True)
    # the network's depth is not exactly affine in inverse depth: the fitted sheet still bows by ~1.5 cm
    # towards the near edge, which read as a flat phantom object. Its known-empty pixels give the bow;
    # a quadratic surface in (x, y) fitted to them is subtracted from every point.
    T = points(cam, Z, table & ~edge)
    if len(T) > 500:
        A = lambda Q: np.c_[np.ones(len(Q)), Q[:, 0], Q[:, 1], Q[:, 0] ** 2, Q[:, 0] * Q[:, 1], Q[:, 1] ** 2]
        keep = np.ones(len(T), bool)
        for _ in range(3):
            c, *_ = np.linalg.lstsq(A(T[keep]), T[keep, 2], rcond=None)
            r = np.abs(A(T) @ c - T[:, 2])
            keep = r <= np.percentile(r, 85)
        P[:, 2] -= A(P) @ c
        fit["bow_mm"] = float(np.ptp(A(T) @ c) * 1000)
    H, lo, _ = heightmap(P, zone)
    objs = objects(H, lo, zone)
    objs, H = drop_hidden(objs, H, lo, cam.C)
    res = {"fit": fit, "inference_s": round(t_inf, 2), "objects": objs, "H": H, "lo": lo, "Z": Z}
    if save:
        _save_views(frame, rel, H, objs, lo, key)
    return res


STABLE_M = 0.006  # m: an object's centre and height may spread this much over the frames and still be real


def analyse_multi(frames: list, cam, key: str = "cam1", ignore: np.ndarray | None = None) -> dict:
    """`analyse` over several frames of a still scene, keeping only what stays put. Next to a translucent
    wash bottle the network conjured a 2-6 cm 'object' in every frame - but it wandered 2.7 cm and its
    height 3.7 cm between frames, while the bottle held within 3 mm. An object must appear in all but one
    frame with its centre and height within STABLE_M (std); the rest is dropped, with its height-map cells.
    The height map is the per-cell median over the frames."""
    runs = [analyse(f, cam, key, ignore, save=(i == 0)) for i, f in enumerate(frames)]
    if len(runs) == 1:
        return runs[0]
    clusters: list[list[dict]] = []
    for r in runs:
        for o in r["objects"]:
            for c in clusters:
                if np.hypot(*(np.array(o["xy"]) - np.array(c[0]["xy"]))) < 0.03:
                    c.append(o)
                    break
            else:
                clusters.append([o])
    keep, dropped = [], []
    for c in clusters:
        xy = np.array([o["xy"] for o in c])
        h = np.array([o["height"] for o in c])
        spread = max(float(np.linalg.norm(xy.std(axis=0))), float(h.std()))
        if len(c) >= len(runs) - 1 and spread <= STABLE_M:
            best = dict(c[int(np.argmin(np.abs(h - np.median(h))))])  # the frame nearest the median height
            best["xy"] = tuple(float(v) for v in np.median(xy, axis=0))
            best["height"] = float(np.median(h))
            best["frames"], best["spread_mm"] = len(c), round(spread * 1000, 1)
            keep.append(best)
        else:
            dropped.append({"xy": tuple(float(v) for v in xy.mean(axis=0)), "frames": len(c),
                            "spread_mm": round(spread * 1000, 1), "height": float(np.median(h))})
    H = np.nanmedian(np.stack([r["H"] for r in runs]), axis=0)
    lo = runs[0]["lo"]
    up = (np.nan_to_num(H, nan=0.0) > MIN_RISE).astype(np.uint8)
    n, lab = cv2.connectedComponents(up, connectivity=8)
    for k in range(1, n):  # a raised patch that is no kept object's is not trusted: no data there
        ij = np.argwhere(lab == k)
        cxy = lo + (ij + 0.5) * CELL
        # kept only if a stable object's centre lies in the patch itself (1.5 cm): patches that merely sit
        # next to one (a phantom 8 cm from a wash bottle) are not
        if not any(np.min(np.hypot(*(cxy - np.array(o["xy"])).T)) < 0.015 for o in keep):
            H[lab == k] = np.nan
    out = dict(runs[0])
    out.update(objects=keep, H=H, dropped=dropped, frames=len(runs),
               inference_s=round(sum(r["inference_s"] for r in runs), 2))
    return out


def _save_views(frame, rel, H, objs, lo, key, out: Path | None = None) -> None:
    out = out or pp_captures() / f"depth_{key}.jpg"
    r = cv2.normalize(rel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    left = cv2.resize(cv2.applyColorMap(r, cv2.COLORMAP_INFERNO), (640, 360))
    hm = np.nan_to_num(H, nan=0.0)
    hv = cv2.applyColorMap(np.clip(hm / 0.15 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    hv[np.isnan(H)] = 40
    hv = cv2.flip(cv2.transpose(hv), -1)  # robot +x up the image, +y to the left, like the map's top view
    hv = cv2.resize(hv, (int(360 * hv.shape[1] / hv.shape[0]), 360), interpolation=cv2.INTER_NEAREST)
    s = 360 / H.shape[0]
    for o in objs:
        cx = hv.shape[1] - 1 - (o["xy"][1] - lo[1]) / CELL * s
        cy = 360 - 1 - (o["xy"][0] - lo[0]) / CELL * s
        cv2.putText(hv, f"{o['length'] * 100:.1f}x{o['width'] * 100:.1f}x{o['height'] * 100:.1f}",
                    (int(cx) - 45, int(cy)), 0, 0.4, (255, 255, 255), 1)
    cv2.imwrite(str(out), np.hstack([left, hv]))


def pp_captures() -> Path:
    p = HERE / "captures"
    p.mkdir(exist_ok=True)
    return p


if __name__ == "__main__":
    import argparse

    sys.path.insert(0, str(HERE))
    import pick_place as pp

    ap = argparse.ArgumentParser()
    ap.add_argument("--frame")
    ap.add_argument("--cam", default="cam1")
    a = ap.parse_args()
    cams = pp.load_cameras()
    f = cv2.imread(a.frame) if a.frame else pp.grab(int(a.cam[3:]))
    r = analyse(f, cams[a.cam], a.cam)
    print(f"inference {r['inference_s']} s; table fit on {r['fit']['pixels']} px, rms {r['fit']['table_rms_mm']:.1f} mm")
    for o in r["objects"]:
        print(f"  object at ({o['xy'][0]:+.3f}, {o['xy'][1]:+.3f}): {o['length'] * 100:.1f} x {o['width'] * 100:.1f}"
              f" x {o['height'] * 100:.1f} cm, long side at {np.degrees(o['angle']):+.0f} deg ({o['cells']} cells)")
    print(f"views -> captures/depth_{a.cam}.jpg")
