"""The gripper's tactile skin (WowSkin, USB serial) next to the gripper motor's own state.

    python tactile.py --probe                    # find the skin's port, print raw lines, channel count, rate
    python tactile.py --report CSV [--out PNG]   # correlate a logged pick: skin vs gripper motor

During `pick_place.py --run --tactile` a `GripLog` samples, on one clock at ~100 Hz:

    t  phase  grip_cmd  grip_pos  grip_vel  grip_eff  skin_mag  skin_0 .. skin_N

where grip_cmd is the commanded opening (0 closed .. 1 open), grip_pos/vel/eff the gripper motor's measured
position, velocity and effort (torque) from i2rt, and skin_mag the norm of the skin's change from its
no-contact baseline (taken while the jaw is still open and touching nothing). The CSV lands next to the
episode in the LeRobot dataset (meta/tactile/episode_NNNNNN.csv), so the dataset schema is unchanged.

The skin's firmware format is not assumed beyond "one sample per line, numbers separated by anything":
every number on a line is a channel. The channel count is learnt from the first lines.
"""

from __future__ import annotations

import csv
import glob
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

PORT_GLOBS = ("/dev/cu.usbmodem*", "/dev/cu.usbserial*", "/dev/cu.wchusbserial*", "/dev/cu.SLAB_USBtoUART*")
NUM = re.compile(rb"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def find_port() -> str:
    ports = sorted(p for g in PORT_GLOBS for p in glob.glob(g))
    if not ports:
        raise RuntimeError("no USB serial device found (looked for " + ", ".join(PORT_GLOBS) + ") - is the skin plugged in?")
    if len(ports) > 1:
        print(f"   (several serial ports: {ports}; using {ports[0]} - pass --tactile-port to choose)")
    return ports[0]


class TactileSkin:
    """Reads the skin on a background thread; `latest()` is the newest channel vector."""

    def __init__(self, port: str | None = None, baud: int = 115200):
        import serial

        self.port = port or find_port()
        self.ser = serial.Serial(self.port, baud, timeout=0.2)
        # the skin's board can sit silent after a reconnect (0 bytes in 4 s on 2026-09-24) until the host
        # toggles DTR and sends a line: wake it on every open
        self.ser.dtr = False
        time.sleep(0.3)
        self.ser.dtr = True
        self.ser.rts = True
        time.sleep(0.3)
        self.ser.write(b"\n")
        self.lock = threading.Lock()
        self.vec: np.ndarray | None = None
        self.n = 0  # samples read
        self.width: int | None = None  # channel count, fixed by the first lines
        self.baseline: np.ndarray | None = None
        self.binary: bool | None = None  # float32 frames (WowSkin) or text lines, decided from the first lines
        self.kinds: list[bool] = []
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _parse(self, line: bytes) -> list[float] | None:
        """One line -> channel values. Text firmware prints numbers; the WowSkin's QT Py M0 sends binary
        frames instead: N little-endian float32 then \\r\\n, as (temperature, x, y, z) per magnetometer.
        The temperatures are dropped: they drift with the room, not with touch."""
        body = line[:-2] if line.endswith(b"\r\n") else line.rstrip(b"\n")
        if self.binary is None and len(self.kinds) < 20:  # decide from the first lines
            self.kinds.append(sum(c < 9 or c > 126 for c in body) > len(body) // 4)
            if len(self.kinds) == 20:
                self.binary = sum(self.kinds) > 10
            return None
        if self.binary:
            if len(body) == 0 or len(body) % 16:  # a frame split by a \\r\\n inside its data: skip, resync
                return None
            f = np.frombuffer(body, "<f4").reshape(-1, 4)
            if not np.all(np.isfinite(f)) or np.any(np.abs(f[:, 0] - 25) > 40):  # temperature column sanity
                return None
            return f[:, 1:].ravel().astype(float).tolist()
        return [float(m) for m in NUM.findall(line)]

    def _loop(self) -> None:
        widths: list[int] = []
        while self.running:
            try:
                line = self.ser.readline()
            except Exception:  # unplugged mid-run: keep the last value, stop reading
                break
            vals = self._parse(line)
            if not vals:
                continue
            if self.width is None:  # the most common count over the first lines (the first is often partial)
                widths.append(len(vals))
                if len(widths) >= 10:
                    self.width = max(set(widths), key=widths.count)
                continue
            if len(vals) != self.width:
                continue
            with self.lock:
                self.vec = np.array(vals)
                self.n += 1

    def wait(self, timeout: float = 5.0) -> None:
        t = time.time()
        while self.vec is None and time.time() - t < timeout:
            time.sleep(0.02)
        if self.vec is None:
            raise RuntimeError(f"no parsable samples from the skin on {self.port} within {timeout:.0f} s")

    def latest(self) -> np.ndarray | None:
        with self.lock:
            return None if self.vec is None else self.vec.copy()

    def tare(self, seconds: float = 0.5) -> np.ndarray:
        """No-contact baseline: the median over `seconds`. Take it while nothing touches the skin."""
        self.wait(12.0)  # the board can take several seconds to start streaming after the port opens
        buf, t = [], time.time()
        while time.time() - t < seconds:
            v = self.latest()
            if v is not None:
                buf.append(v)
            time.sleep(0.01)
        self.baseline = np.median(np.array(buf), axis=0)
        return self.baseline

    def magnitude(self, v: np.ndarray | None = None) -> float:
        v = self.latest() if v is None else v
        if v is None:
            return float("nan")
        b = self.baseline if self.baseline is not None else np.zeros_like(v)
        return float(np.linalg.norm(v - b))

    def close(self) -> None:
        self.running = False
        self.thread.join(timeout=1.0)
        self.ser.close()


class GripLog:
    """Samples the skin and the gripper motor together, on one clock, until stop(); save() writes the CSV."""

    def __init__(self, skin: TactileSkin, get_obs: Callable[[], dict], get_cmd: Callable[[], float],
                 get_phase: Callable[[], str], hz: float = 100.0):
        self.skin, self.get_obs, self.get_cmd, self.get_phase, self.hz = skin, get_obs, get_cmd, get_phase, hz
        self.rows: list[list] = []
        self.running = False
        self.t0 = 0.0

    def start(self) -> None:
        self.t0 = time.monotonic()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        period, i = 1.0 / self.hz, 0
        while self.running:
            target = self.t0 + i * period
            now = time.monotonic()
            if now < target:
                time.sleep(target - now)
            try:
                o = self.get_obs()
                v = self.skin.latest()
                self.rows.append([round(time.monotonic() - self.t0, 4), self.get_phase(), float(self.get_cmd()),
                                  float(o["gripper_pos"][0]), float(o["gripper_vel"][0]), float(o["gripper_eff"][0]),
                                  self.skin.magnitude(v)] + ([] if v is None else [float(x) for x in v]))
            except Exception as e:  # never let logging take the run down
                self.rows.append([round(time.monotonic() - self.t0, 4), f"log error: {e}"])
            i += 1

    def stop(self) -> None:
        self.running = False
        if getattr(self, "thread", None):
            self.thread.join(timeout=1.0)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        width = self.skin.width or 0
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "phase", "grip_cmd", "grip_pos", "grip_vel", "grip_eff", "skin_mag"]
                       + [f"skin_{k}" for k in range(width)])
            w.writerows(r for r in self.rows if len(r) >= 7)
        return path


# ================================================================================ correlation report
def _load(path: Path) -> dict[str, np.ndarray]:
    with Path(path).open() as fh:
        rows = list(csv.reader(fh))
    head, body = rows[0], rows[1:]
    out = {h: np.array([r[i] for r in body]) for i, h in enumerate(head) if h == "phase"}
    for i, h in enumerate(head):
        if h != "phase":
            out[h] = np.array([float(r[i]) if i < len(r) and r[i] not in ("", "nan") else np.nan for r in body])
    return out


def _r(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _onset(t: np.ndarray, x: np.ndarray, mask: np.ndarray, k: float = 5.0) -> float:
    """First time inside `mask` that x rises k robust-sigmas above its level before the mask."""
    pre = x[(t < t[mask][0]) & np.isfinite(x)] if mask.any() else x[np.isfinite(x)]
    if len(pre) < 5 or not mask.any():
        return float("nan")
    med = np.median(pre)
    mad = np.median(np.abs(pre - med)) * 1.4826 + 1e-9
    hit = np.where(mask & (np.abs(x - med) > k * mad))[0]
    return float(t[hit[0]]) if len(hit) else float("nan")


def _half_rise(t: np.ndarray, x: np.ndarray, closing: np.ndarray, ph: np.ndarray) -> float:
    hold = ph == "hold 5 s"
    if not closing.any() or not hold.any():
        return float("nan")
    lo = float(np.median(x[t < t[closing][0]][-100:]))
    hi = float(np.median(x[hold]))
    win = np.where((t >= t[closing][0]) & (t <= t[hold][-1]))[0]
    hit = win[(x[win] - lo) >= 0.5 * (hi - lo)] if hi != lo else []
    return float(t[hit[0]]) if len(hit) else float("nan")


def _median(x: np.ndarray, t: np.ndarray, win: float = 0.3) -> np.ndarray:
    from scipy.ndimage import median_filter

    n = max(3, int(win / max(float(np.median(np.diff(t))), 1e-4)) | 1)
    return median_filter(np.nan_to_num(x, nan=float(np.nanmedian(x))), size=n, mode="nearest")


def report(path: Path, out: Path | None = None) -> dict:
    d = _load(path)
    t, ph = d["t"], d["phase"]
    grip = np.isin(ph, ["close on the object", "lift", "hold 5 s", "look at the scene again",
                        "carry to the place point", "lower"])
    closing = ph == "close on the object"
    eff, pos, mag = d["grip_eff"], d["grip_pos"], d["skin_mag"]
    res = {
        "samples": int(len(t)), "seconds": float(t[-1]) if len(t) else 0.0,
        "r(skin, |effort|) whole run": _r(mag, np.abs(eff)),
        "r(skin, |effort|) while gripping": _r(mag[grip], np.abs(eff[grip])),
        "r(skin, opening) while closing": _r(mag[closing], pos[closing]),
        # contact = the first moment in the closing/hold window at which each signal (0.3 s running median)
        # is halfway from its pre-close level to its level during the hold; the motor chatters while the jaw
        # is still moving, so "first departure from rest" would time the motion, not the squeeze
        "skin contact (s, half of hold level)": _half_rise(t, _median(mag, t), closing, ph),
        "motor squeeze (s, half of hold level)": _half_rise(t, _median(np.abs(eff), t), closing, ph),
    }
    # lag: shift the skin against the effort over +-0.5 s during the grasp, take the best correlation
    if grip.sum() > 20:
        dt = float(np.median(np.diff(t)))
        best = (float("nan"), 0.0)
        for s in range(-int(0.5 / dt), int(0.5 / dt) + 1):
            a, b = np.abs(eff), np.roll(mag, s)
            r = _r(a[grip], b[grip])
            if np.isfinite(r) and (not np.isfinite(best[0]) or r > best[0]):
                best = (r, s * dt)
        res["best r with skin shifted (s)"] = best
    print(f"{path}")
    for k, v in res.items():
        print(f"  {k:<36} {v if not isinstance(v, float) else round(v, 3)}")
    if out is not None:
        _plot(d, grip, out)
        print(f"  plot -> {out}")
    return res


def _plot(d: dict, grip: np.ndarray, out: Path) -> None:
    """Three stacked traces (commanded/measured opening, motor effort, skin magnitude) with phases shaded;
    plain OpenCV so no plotting package is needed."""
    import cv2

    t = d["t"]
    W, H, pad = 1200, 720, 60
    img = np.full((H, W, 3), 255, np.uint8)
    lanes = [("opening (cmd dashed, measured solid)", [d["grip_cmd"], d["grip_pos"]]),
             ("gripper motor effort", [d["grip_eff"]]), ("skin |change from baseline|", [d["skin_mag"]])]
    lh = (H - 2 * pad) // 3
    x = lambda v: (pad + (v - t[0]) / max(t[-1] - t[0], 1e-6) * (W - 2 * pad)).astype(np.int32)
    xs = x(t)
    for i in np.where(grip)[0]:
        cv2.line(img, (xs[i], pad), (xs[i], H - pad), (235, 245, 255), 1)
    ph = d["phase"]
    for i in np.where(np.r_[True, ph[1:] != ph[:-1]])[0]:
        cv2.line(img, (xs[i], pad), (xs[i], H - pad), (200, 200, 200), 1)
        cv2.putText(img, str(ph[i])[:18], (xs[i] + 2, pad - 8 - 12 * (i % 3)), 0, 0.33, (90, 90, 90), 1)
    colors = [(40, 40, 200), (200, 90, 30), (30, 150, 30)]
    for li, (name, series) in enumerate(lanes):
        y0 = pad + li * lh
        cv2.rectangle(img, (pad, y0), (W - pad, y0 + lh - 10), (180, 180, 180), 1)
        cv2.putText(img, name, (pad + 5, y0 + 15), 0, 0.45, (0, 0, 0), 1)
        allv = np.concatenate([s[np.isfinite(s)] for s in series]) if series else np.array([0.0])
        lo, hi = (float(allv.min()), float(allv.max())) if len(allv) else (0.0, 1.0)
        hi = hi if hi > lo else lo + 1.0
        for si, s in enumerate(series):
            m = np.isfinite(s)
            ys = (y0 + lh - 15 - (s - lo) / (hi - lo) * (lh - 30)).astype(np.int32)
            pts = np.c_[xs[m], ys[m]].reshape(-1, 1, 2)
            if len(pts) > 1:
                if li == 0 and si == 0:  # commanded: dashed
                    for a in range(0, len(pts) - 1, 6):
                        cv2.polylines(img, [pts[a:a + 3]], False, colors[li], 1)
                else:
                    cv2.polylines(img, [pts], False, colors[li], 2)
        cv2.putText(img, f"{lo:.3g}..{hi:.3g}", (W - pad - 120, y0 + 15), 0, 0.4, (90, 90, 90), 1)
    cv2.imwrite(str(out), img)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--report")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.report:
        report(Path(a.report), Path(a.out) if a.out else Path(a.report).with_suffix(".png"))
    elif a.probe:
        import serial

        port = a.port or find_port()
        print(f"port {port} @ {a.baud}")
        with serial.Serial(port, a.baud, timeout=0.5) as s:
            for _ in range(8):
                print("  raw:", s.readline()[:160])
        sk = TactileSkin(port, a.baud)
        sk.wait()
        n0, t0 = sk.n, time.time()
        time.sleep(2.0)
        print(f"channels {sk.width}, {(sk.n - n0) / (time.time() - t0):.0f} samples/s, latest {np.round(sk.latest(), 2)}")
        sk.tare()
        print("press the skin now...")
        for _ in range(20):
            print(f"  |change| {sk.magnitude():8.2f}")
            time.sleep(0.25)
        sk.close()
