"""Find an upright cylinder (a clear vial, a can, ...) near a guess in the detection camera, by where the live frame
differs from the empty-zone background (captures/bg_cam1.png, kept aligned by calib_tools/cam_rebump.py): the
cylinder's projected silhouette (known diameter and height) is slid over that difference and the best overlap kept.
Works for clear objects too - they only need to look different from the empty table, not a colour.

    python scripts/cyl_find.py x,y diameter height [--show out.jpg]   ->  "x,y,score" or nothing (not found)
    from cyl_find import find; find(frame, bg, cam, (x, y), d, h) -> ((x, y), score) or (None, score)
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, "/Users/solotech/yam")
MIN_SCORE = 0.45  # overlap (IoU) below this: not found
SEARCH = 0.07  # m around the guess


def silhouette(cam, x, y, r, h, shape):
    th = np.linspace(0, 2 * np.pi, 48)
    P = np.vstack([np.c_[x + r * np.cos(th), y + r * np.sin(th), np.full(48, z)] for z in (0.0, h)])
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(cam.project(P).astype(np.int32)), 1)
    return m


def diff_mask(frame, bg):
    """Where the scene differs from the empty background in fine detail. Each image has its broad brightness removed
    first (minus a wide blur, channel by channel), so a change of exposure or daylight - which moves every pixel -
    does not count; an object's edges, highlights and shadow outline do."""
    def detail(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        return lab - cv2.GaussianBlur(lab, (0, 0), 15)
    d = np.abs(detail(frame) - detail(bg))
    m = ((d[..., 0] > 14) | (d[..., 1] > 6) | (d[..., 2] > 6)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def find(frame, bg, cam, guess, d, h):
    return _fit(diff_mask(frame, bg), cam, guess, d, h)


def _solid(D, cam, guess, d, h, reach=0.045):
    """A clear object only differs at its edges, cap and base: keep the changed pixels within `reach` of the guess
    (projected), and fill them in as one convex blob - the object's silhouette."""
    near = silhouette(cam, guess[0], guess[1], d / 2 + reach, h + 0.02, D.shape)
    pts = cv2.findNonZero(D & near)
    out = np.zeros_like(D)
    if pts is not None and len(pts) > 30:
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out


def _fit(D, cam, guess, d, h):
    r = d / 2
    g = np.asarray(guess, float)
    D = _solid(D, cam, g, d, h)

    ring = np.ones((25, 25), np.uint8)

    def score(x, y):
        """Overlap of the silhouette with the difference, penalised by difference right around it (a silhouette
        sitting inside a bigger blob - the object's shadow, a neighbour - does not score as well as the true fit)."""
        s = silhouette(cam, x, y, r, h, D.shape)
        inter = int((s & D).sum())
        around = int(((D > 0) & (cv2.dilate(s, ring) > 0) & (s == 0)).sum())
        return inter / (int(s.sum()) + around + 1e-9)

    best = (-1.0, g[0], g[1])
    for step, span in ((0.006, SEARCH), (0.002, 0.008), (0.0007, 0.0025)):
        cx, cy = best[1], best[2]
        for x in np.arange(cx - span, cx + span + 1e-9, step):
            for y in np.arange(cy - span, cy + span + 1e-9, step):
                sc = score(x, y)
                if sc > best[0]:
                    best = (sc, x, y)
    sc, x, y = best
    return ((float(x), float(y)) if sc >= MIN_SCORE else None), float(sc)


def align(ref, frame):
    """ref warped onto frame by a homography fitted to background feature matches (the table is a plane; a camera
    that crept a little between the two shots is undone)."""
    orb = cv2.ORB_create(4000)
    ka, da = orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
    kb, db = orb.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    m = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db), key=lambda z: z.distance)[:1500]
    A = np.float32([ka[z.queryIdx].pt for z in m]); B = np.float32([kb[z.trainIdx].pt for z in m])
    H, _ = cv2.findHomography(A, B, cv2.RANSAC, 2.0)
    return cv2.warpPerspective(ref, H, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def find_moved(before, after, cam, guess, d, h, exclude=None):
    """Where the cylinder is in `after`, from what changed since `before` (a frame taken just before the move, with
    the object elsewhere). `exclude`: where it stood in `before` (that spot changes too) - ignored."""
    ref = align(before, after)
    D = diff_mask(after, ref)
    if exclude is not None:
        D &= 1 - silhouette(cam, exclude[0], exclude[1], d / 2 + 0.02, h + 0.01, D.shape)
    return _fit(D, cam, guess, d, h)


if __name__ == "__main__":
    from calib3d import load_cameras
    from scene3d import BACKGROUND, grab

    g = [float(v) for v in sys.argv[1].split(",")]
    d, h = float(sys.argv[2]), float(sys.argv[3])
    cam = load_cameras()["cam1"]
    frame, bg = grab(1), cv2.imread(str(BACKGROUND["cam1"]))
    xy, sc = find(frame, bg, cam, g, d, h)
    print(f"score {sc:.2f}", file=sys.stderr)
    if "--show" in sys.argv:
        v = frame.copy()
        if xy is not None:
            s = silhouette(cam, xy[0], xy[1], d / 2, h, frame.shape[:2])
            cv2.drawContours(v, cv2.findContours(s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 0, 255), 2)
        cv2.imwrite(sys.argv[sys.argv.index("--show") + 1], v)
    if xy is not None:
        print(f"{xy[0]:.4f},{xy[1]:.4f},{sc:.2f}")
