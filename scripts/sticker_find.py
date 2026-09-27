"""Find an object by a coloured sticker on its top (default yellow) in the detection camera: the sticker blob nearest
the guess, its centroid back-projected onto the plane of the object's top (height H) gives the object's axis.
Sharp and colour-based, so a clear object (the glass vial) is found as reliably as an opaque one.

    python scripts/sticker_find.py x,y diameter height [--show out.jpg] [--hue 18,38]  ->  "x,y,area" or nothing
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, "/Users/solotech/yam")
SEARCH = 0.08  # m: only a sticker whose axis lands this close to the guess counts
MIN_AREA = 25  # px


def find(frame, cam, guess, h, hue=(22, 40)):  # a pale sticker in bright light reads S~50: keep the saturation cut low
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] >= hue[0]) & (hsv[..., 0] <= hue[1]) & (hsv[..., 1] > 22) & (hsv[..., 2] > 170)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    best = None
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        xy = cam.hit_plane(float(cen[i][0]), float(cen[i][1]), h)[:2]
        dist = float(np.linalg.norm(xy - np.asarray(guess, float)))
        if dist <= SEARCH and (best is None or dist < best[0]):
            best = (dist, xy, int(st[i, cv2.CC_STAT_AREA]), cen[i])
    return best


if __name__ == "__main__":
    from calib3d import load_cameras
    from scene3d import grab

    g = [float(v) for v in sys.argv[1].split(",")]
    h = float(sys.argv[3])
    hue = tuple(int(v) for v in sys.argv[sys.argv.index("--hue") + 1].split(",")) if "--hue" in sys.argv else (18, 38)
    cam = load_cameras()["cam1"]
    frame = grab(1)
    b = find(frame, cam, g, h, hue)
    if "--show" in sys.argv:
        v = frame.copy()
        if b is not None:
            cv2.drawMarker(v, (int(b[3][0]), int(b[3][1])), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
            th = np.linspace(0, 2 * np.pi, 48)
            r = float(sys.argv[2]) / 2
            for z in (0.0, h):
                P = np.c_[b[1][0] + r * np.cos(th), b[1][1] + r * np.sin(th), np.full(48, z)]
                cv2.polylines(v, [cam.project(P).astype(np.int32)], True, (0, 255, 0), 2)
        cv2.imwrite(sys.argv[sys.argv.index("--show") + 1], v)
    if b is None:
        print("no sticker near the guess", file=sys.stderr)
    else:
        print(f"{b[1][0]:.4f},{b[1][1]:.4f},{b[2]}")
