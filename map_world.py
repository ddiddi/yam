"""The sensors and the depth world for the workspace map: what live_map.build() embeds next to scene.json.

    python map_world.py        # camera + serial reads only (no arm): -> captures/map_world.json

sensors  every camera (calibrated? how far it has moved since its background, resolution), the depth
         model's table fit, the tactile skin (port, channels, rate) with the latest skin-vs-motor report,
         and the arm's fingertip model offset from the last calibration sweep
world    the first-scene-analysis height map (Depth Anything V2 Small on the detection camera, 5 mm cells,
         mm heights) and every object in it with L x W x H and its long side's direction
"""

from __future__ import annotations

import glob
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "captures" / "map_world.json"


def _cameras() -> list[dict]:
    import pick_place as pp

    cal = pp.load_cameras()
    out = []
    for i in range(3):
        key = f"cam{i}"
        try:
            f = pp.grab(i)
        except Exception as e:
            out.append({"name": key, "status": "offline", "note": str(e)[:80]})
            continue
        shift = float(pp.camera_shift(key, f)) if pp.BACKGROUND[key].exists() else float("nan")
        role = "detection + depth" if key == pp.DET else ("fused (carving)" if key in cal else "recording only")
        ok = key in cal and np.isfinite(shift) and shift <= pp.CAM_MOVED_PX
        status = ("calibrated" if ok else "moved - excluded until recalibrated") if key in cal else "not calibrated"
        out.append({"name": key, "role": role, "status": status, "shift_px": round(shift, 1),
                    "resolution": f"{f.shape[1]}x{f.shape[0]}", "calibrated": key in cal})
    return out


def _depth(frame) -> tuple[dict, dict]:
    import depth
    import pick_place as pp

    cams = pp.load_cameras()
    r = depth.analyse(frame, cams[pp.DET], pp.DET)
    H = r["H"]
    grid = [[None if not np.isfinite(v) else int(round(v * 1000)) for v in row] for row in H]
    sensor = {"name": "Depth Anything V2 Small", "runtime": "ONNX Runtime (CPU)", "camera": pp.DET,
              "inference_s": r["inference_s"], "table_fit_rms_mm": round(r["fit"]["table_rms_mm"], 1),
              "table_bow_mm": round(r["fit"].get("bow_mm", float("nan")), 1), "cells": int(np.isfinite(H).sum())}
    objs = [{"xy": [round(o["xy"][0], 4), round(o["xy"][1], 4)], "length": round(o["length"] + pp.DEPTH_EDGE, 4),
             "width": round(o["width"] + pp.DEPTH_EDGE, 4), "height": round(o["height"], 4),
             "angle": round(o["angle"], 4), "shape": o.get("shape", "")} for o in r["objects"]]
    world = {"cell": depth.CELL, "lo": [float(r["lo"][0]), float(r["lo"][1])], "H_mm": grid,
             "min_rise_mm": int(depth.MIN_RISE * 1000), "objects": objs,
             "measured": time.strftime("%Y-%m-%d %H:%M:%S")}
    return sensor, world


def _skin() -> dict:
    ports = sorted(glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/cu.usbserial*"))
    out = {"name": "WowSkin tactile skin", "board": "QT Py M0 (USB serial)", "status": "not connected"}
    if ports:
        try:
            from tactile import TactileSkin

            sk = TactileSkin(ports[0])
            sk.wait(6)
            n0, t0 = sk.n, time.time()
            time.sleep(1.0)
            out.update(status="streaming", port=ports[0], channels=sk.width,
                       rate_hz=int((sk.n - n0) / (time.time() - t0)),
                       layout="5 magnetometers x (x, y, z)")
            sk.close()
        except Exception as e:
            out.update(status=f"port present, no data ({str(e)[:60]})", port=ports[0])
    logs = sorted(glob.glob(str(HERE / "datasets" / "*" / "meta" / "tactile" / "episode_*.csv")),
                  key=lambda p: Path(p).stat().st_mtime)
    for p in reversed(logs):  # the latest log that has a grasp in it
        try:
            from tactile import report

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = report(Path(p), None)
            if np.isfinite(r.get("r(skin, |effort|) while gripping", float("nan"))):
                out["last_grasp"] = {"log": str(Path(p).relative_to(HERE)),
                                     "r_skin_effort_gripping": round(r["r(skin, |effort|) while gripping"], 3),
                                     "r_skin_opening_closing": round(r["r(skin, opening) while closing"], 3),
                                     "skin_contact_s": round(r["skin contact (s, half of hold level)"], 2),
                                     "motor_squeeze_s": round(r["motor squeeze (s, half of hold level)"], 2)}
                break
        except Exception:
            continue
    return out


def _arm() -> dict:
    """The fingertip model offset measured by the latest calibration sweep (cam1 vs FK, clean detections)."""
    import pick_place as pp
    from calib3d import load_cameras

    sweeps = sorted(glob.glob(str(HERE / "captures" / "sweep_zone*" / "sweep.json")), key=lambda p: Path(p).stat().st_mtime)
    if not sweeps:
        return {}
    c = load_cameras()[pp.DET]
    errs = []
    for r in json.loads(Path(sweeps[-1]).read_text()):
        uv = r["px"].get(pp.DET)
        if uv:
            e = float(np.linalg.norm(np.array(uv) - c.project(np.array(r["fk"])[None])[0]))
            if e < 40:
                errs.append(e)
    return {"name": "YAM arm (i2rt, CAN)", "sweep": str(Path(sweeps[-1]).parent.name),
            "sweep_points": len(errs), "fingertip_model_px": round(float(np.median(errs)), 1) if errs else None,
            "note": "visual check measures ~1.3 cm model offset per run and corrects it"}


def build() -> dict:
    sys.path.insert(0, str(HERE))
    import pick_place as pp

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        cams = _cameras()
        dsensor, world = _depth(pp.grab(int(pp.DET[3:])))
        data = {"sensors": {"cameras": cams, "depth": dsensor, "skin": _skin(), "arm": _arm()}, "world": world}
    OUT.write_text(json.dumps(data))
    return data


if __name__ == "__main__":
    d = build()
    s = d["sensors"]
    for c in s["cameras"]:
        print(f"  {c['name']}: {c.get('status')} (shift {c.get('shift_px')} px, {c.get('role', '')})")
    print(f"  depth: rms {s['depth']['table_fit_rms_mm']} mm, {s['depth']['inference_s']} s; "
          f"{len(d['world']['objects'])} object(s)")
    print(f"  skin: {s['skin']['status']} {s['skin'].get('channels', '')} ch {s['skin'].get('rate_hz', '')} Hz; "
          f"last grasp {s['skin'].get('last_grasp')}")
    print(f"  arm: {s['arm']}")
    print(f"-> {OUT}")
