"""Read the Amazon Basics shipping scale's LCD (lb + oz mode) from a camera frame (cam2 faces it).

    python scale_read.py [image ...]            # read saved frames
    python scale_read.py --live [--cam 2]       # keep reading the camera: "oz  (lb, oz)" per line

read(frame) -> (total_oz, (lb, oz), debug) or None. The display is found as the pale blue-white backlit rectangle
framed by the scale's black body, rectified, deslanted (the digits are italic), split into digit blobs (the
separate segments of one digit are joined), and each digit decoded from its 7 segments. The first digit is the
pounds; the rest are ounces with one decimal (d = 0.1 oz = 2.8 g).
"""
from __future__ import annotations

import sys
import time

import cv2
import numpy as np

W, H = 360, 120  # the rectified display
SLANT = 0.20  # the digits' italic lean (x per y), undone before decoding
CELLS = (122, 183, 236, 290)  # left x of the pounds, ounce-tens, ounce-units and ounce-tenths digit cells

# segment sample points (x, y) in a digit's box, 0..1: a top, b top-right, c bottom-right, d bottom, e bottom-left,
# f top-left, g middle
SEGS = {"a": (0.50, 0.08), "b": (0.86, 0.28), "c": (0.86, 0.72), "d": (0.50, 0.92),
        "e": (0.14, 0.72), "f": (0.14, 0.28), "g": (0.50, 0.50)}
DIGITS = {"abcdef": 0, "bc": 1, "abdeg": 2, "abcdg": 3, "bcfg": 4, "acdfg": 5, "acdefg": 6, "abc": 7,
          "abcdefg": 8, "abcdfg": 9, "abcfg": 9}  # this LCD draws its 9 without the bottom segment


def find_lcd(frame: np.ndarray):
    """The display's 4 corners (tl, tr, br, bl) in the frame, or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dark = (hsv[..., 2] < 70).astype(np.uint8)
    best = None
    for vmin, smax in ((170, 90), (150, 120), (130, 140)):
        m = ((hsv[..., 2] > vmin) & (hsv[..., 1] < smax)).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            a = cv2.contourArea(c)
            if a < 2000:
                continue
            r = cv2.minAreaRect(c)
            w, h = r[1]
            ar, fill = max(w, h) / max(min(w, h), 1), a / (w * h + 1e-9)
            if not (1.8 < ar < 4.5 and fill > 0.75):
                continue
            # framed by the black body: most of a ring just outside it is dark
            ring = np.zeros(m.shape, np.uint8)
            cv2.drawContours(ring, [cv2.boxPoints(((r[0][0], r[0][1]), (w * 1.25, h * 1.5), r[2])).astype(np.int32)], -1, 255, -1)
            cv2.drawContours(ring, [c], -1, 0, -1)
            framed = dark[ring > 0].mean() if (ring > 0).any() else 0
            if framed > 0.4 and (best is None or a > best[0]):
                best = (a, r)
        if best is not None:
            break
    if best is None:
        return None
    box = cv2.boxPoints(best[1])
    # the display is wider than tall: order the corners along its long side
    c = box.mean(0)
    ang = np.arctan2(box[:, 1] - c[1], box[:, 0] - c[0])
    box = box[np.argsort(ang)]  # clockwise from top-left-ish (image y down)
    w01 = np.linalg.norm(box[1] - box[0])
    w12 = np.linalg.norm(box[2] - box[1])
    if w12 > w01:
        box = np.roll(box, -1, axis=0)
    # make box[0] the top-left: of the two long sides, the upper one first
    if box[0][1] + box[1][1] > box[2][1] + box[3][1]:
        box = np.roll(box, 2, axis=0)
    if box[0][0] > box[1][0]:
        box = box[[1, 0, 3, 2]]
    return box.astype(np.float32)


def rectify(frame, box):
    M = cv2.getPerspectiveTransform(box, np.float32([[0, 0], [W, 0], [W, H], [0, H]]))
    g = cv2.cvtColor(cv2.warpPerspective(frame, M, (W, H)), cv2.COLOR_BGR2GRAY)
    # undo the italic lean: x' = x - SLANT * (H/2 - y)  (tops lean right)
    S = np.float32([[1, SLANT, -SLANT * H / 2], [0, 1, 0]])
    return cv2.warpAffine(g, S, (W, H), borderMode=cv2.BORDER_REPLICATE)


def decode_digit(ink: np.ndarray, fixed: bool = False) -> int | None:
    h, w = ink.shape
    if not fixed and w < 0.42 * h:  # a narrow blob: the right segments only
        return 1
    on = set()
    for s, (fx, fy) in SEGS.items():
        x, y = int(fx * (w - 1)), int(fy * (h - 1))
        win = ink[max(0, y - 4):y + 5, max(0, x - 4):x + 5]
        if win.size and win.mean() > 0.30:
            on.add(s)
    # nearest 7-segment pattern (one segment may misread: this LCD draws the 0's bottom with a notch)
    best = min(((len(on ^ set(k)), v) for k, v in DIGITS.items()), key=lambda t: t[0])
    return best[1] if best[0] <= 1 else None


def read(frame: np.ndarray):
    box = find_lcd(frame)
    if box is None:
        return None
    g = rectify(frame, box)
    # the segments are the dark ink on the backlight
    ink = (g < cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]).astype(np.uint8)
    ink[:, :int(0.06 * W)] = 0  # the frame's edges
    ink[:, int(0.94 * W):] = 0
    ink[:int(0.04 * H)] = 0
    ink[int(0.96 * H):] = 0
    # the display's layout is fixed: pounds, then ounces tens / units / tenths, each cell 42 px of the rectified
    # 360 x 120 image (measured on frames from two camera poses). A cell with no segment lit is blank.
    vals = []
    for x0 in CELLS:
        cell = ink[8:112, x0:x0 + 42]
        vals.append(None if cell.mean() < 0.03 else decode_digit(cell, fixed=True))
        if cell.mean() >= 0.03 and vals[-1] is None:
            return None
    lb_d, tens, units, tenths = vals
    if lb_d is None or units is None or tenths is None:
        return None
    digits = [(0, lb_d)]
    lb, oz = lb_d, (tens or 0) * 10 + units + tenths / 10.0
    if oz >= 16.0:
        return None
    return lb * 16.0 + oz, (lb, oz), g


if __name__ == "__main__":
    if "--live" in sys.argv:
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from lerobot_record import LiveCamera

        cam = LiveCamera(int(sys.argv[sys.argv.index("--cam") + 1]) if "--cam" in sys.argv else 2)
        time.sleep(1.0)
        while True:
            r = read(cam.latest())
            print(f"{time.time():.2f} " + ("--" if r is None else f"{r[0]:.1f} oz  ({r[1][0]} lb {r[1][1]:.1f} oz)"), flush=True)
            time.sleep(0.2)
    for p in sys.argv[1:]:
        r = read(cv2.imread(p))
        print(p, "->", "unreadable" if r is None else f"{r[1][0]} lb {r[1][1]:.1f} oz")
