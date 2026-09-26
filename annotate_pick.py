"""Mark an object and where the gripper should grab it, by hand, on the detection camera (cam1). Writes a pick file
that `pick_place.py --pick-json FILE` runs from.

    YAM_CAM_MAP="0:0,1:1,2:3,3:2" python annotate_pick.py NAME        # -> captures/picks/NAME.json

In the window (live camera image):
  b        bounds mode (default): left-click the object's outline WHERE IT MEETS THE TABLE, one corner/edge point per
           click; the polygon closes itself
  g        grasp mode: left-click the two spots the two fingertips should touch (at the grasp height)
  u        undo the last click of the current mode        r   re-grab the camera frame
  Enter/s  save and quit                                   q / Esc   quit without saving
Trackbars: the object's height (mm) and the grasp height (mm, where the fingertips close). The overlay redraws live:
the bounds as a box from the table to the object height (green), the two fingertip points and the jaw line (red), and
the jaw width. Keep the width under ~85 mm (the jaw opens to 95 mm).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from calib3d import load_cameras
from scene3d import grab

HERE = Path(__file__).resolve().parent
OUT = HERE / "captures" / "picks"
WIN = "annotate pick (b: bounds, g: grasp, u: undo, r: refresh, Enter: save, q: quit)"


def main(name: str) -> None:
    cam = load_cameras()["cam1"]
    frame = grab(1)
    bounds_px: list[tuple[int, int]] = []
    grasp_px: list[tuple[int, int]] = []
    mode = ["b"]
    last = [None]

    def click(ev, x, y, *_):
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        if mode[0] == "b":
            bounds_px.append((x, y))
        elif len(grasp_px) < 2:
            grasp_px.append((x, y))

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    cv2.setMouseCallback(WIN, click)
    cv2.createTrackbar("object height mm", WIN, 50, 300, lambda v: None)
    cv2.createTrackbar("grasp height mm", WIN, 20, 200, lambda v: None)

    def world():
        h = cv2.getTrackbarPos("object height mm", WIN) / 1000.0
        zg = cv2.getTrackbarPos("grasp height mm", WIN) / 1000.0
        B = [cam.hit_plane(float(u), float(v), 0.0)[:2] for u, v in bounds_px]
        G = [cam.hit_plane(float(u), float(v), zg)[:2] for u, v in grasp_px]
        return h, zg, B, G

    while True:
        h, zg, B, G = world()
        v = frame.copy()
        if len(B) >= 2:
            for z in (0.0, h):
                P = np.c_[np.array(B), np.full(len(B), z)]
                cv2.polylines(v, [cam.project(P).astype(np.int32)], len(B) > 2, (0, 255, 0), 2)
        for u, w in bounds_px:
            cv2.circle(v, (u, w), 4, (0, 255, 0), -1)
        for u, w in grasp_px:
            cv2.drawMarker(v, (u, w), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        msg = f"mode: {'BOUNDS' if mode[0] == 'b' else 'GRASP'}  |  height {h * 1000:.0f} mm, grasp at {zg * 1000:.0f} mm"
        if len(G) == 2:
            cv2.line(v, grasp_px[0], grasp_px[1], (0, 0, 255), 2)
            width = float(np.linalg.norm(G[1] - G[0]))
            msg += f"  |  jaw width {width * 1000:.0f} mm" + ("  (TOO WIDE)" if width > 0.085 else "")
        cv2.putText(v, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(WIN, v)
        state = (tuple(bounds_px), tuple(grasp_px), round(h, 4), round(zg, 4))
        if state != last[0]:  # a draft on every change, so the marks can be read (and refined) before saving
            last[0] = state
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / f"{name}.draft.json").write_text(json.dumps(
                {"name": name, "bounds_px": bounds_px, "grasp_px": grasp_px, "height": h, "grasp_z": zg}))
            cv2.imwrite(str(OUT / f"{name}.draft.jpg"), v)
        k = cv2.waitKey(50) & 0xFF
        if k in (ord("b"), ord("g")):
            mode[0] = chr(k)
        elif k == ord("u"):
            lst = bounds_px if mode[0] == "b" else grasp_px
            if lst:
                lst.pop()
        elif k == ord("r"):
            frame = grab(1)
        elif k in (ord("q"), 27):
            print("quit without saving")
            break
        elif k in (13, 10, ord("s")):
            if len(bounds_px) < 3 or len(grasp_px) != 2:
                print("need at least 3 bounds points and exactly 2 grasp points")
                continue
            h, zg, B, G = world()
            c = (G[0] + G[1]) / 2
            f = (G[1] - G[0]) / (np.linalg.norm(G[1] - G[0]) + 1e-9)
            doc = {
                "name": name, "camera": "cam1", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "object": {"bounds_px": bounds_px, "bounds_xy": np.round(B, 4).tolist(), "height": round(h, 4),
                           "bounds_centre": np.round(np.mean(B, axis=0), 4).tolist()},
                "grasp": {"points_px": grasp_px, "points_xy": np.round(G, 4).tolist(), "z": round(zg, 4),
                          "centre": np.round(c, 4).tolist(), "width": round(float(np.linalg.norm(G[1] - G[0])), 4),
                          "close_dir": np.round(f, 4).tolist(),
                          # pick_place --jaw-across takes the direction the jaw closes ACROSS (a handle's axis)
                          "jaw_across": np.round([-f[1], f[0]], 4).tolist()},
            }
            OUT.mkdir(parents=True, exist_ok=True)
            p = OUT / f"{name}.json"
            p.write_text(json.dumps(doc, indent=1))
            cv2.imwrite(str(OUT / f"{name}.jpg"), v)
            print(f"saved {p}: grasp centre {doc['grasp']['centre']}, width {doc['grasp']['width'] * 1000:.0f} mm, "
                  f"at z {zg * 1000:.0f} mm; object height {h * 1000:.0f} mm")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else time.strftime("pick_%H%M%S"))
