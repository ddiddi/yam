"""The two white labels on the scale's black top as a calibration target: 8 corners, ordered the same way in every
view. The long label ("Peak Technology Enterprises Inc") gives the u axis (its long side), the short one ("Property of")
lies to its +v side; corners are listed per label as (-u,-v), (+u,-v), (+u,+v), (-u,+v), long label first.

    from plate_labels import label_corners; label_corners(frame) -> (8, 2) float or None
"""
import cv2
import numpy as np


def _quads(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dark = (hsv[..., 2] < 120).astype(np.uint8)
    white = ((hsv[..., 2] > 140) & (hsv[..., 1] < 60)).astype(np.uint8)
    bluish = ((hsv[..., 0] > 80) & (hsv[..., 0] < 115) & (hsv[..., 1] > 25)).astype(np.uint8)  # the LCD backlight
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 5x5 merged the two labels in a far view
    out = []
    for c in cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        a = cv2.contourArea(c)
        if a < 150:
            continue
        r = cv2.minAreaRect(c)
        w, h = r[1]
        if min(w, h) < 4:
            continue
        ar = max(w, h) / min(w, h)
        fill = a / (w * h + 1e-9)
        if not (3.0 < ar < 25 and fill > 0.6):
            continue
        # a label sits on the black top: most of a ring around it is dark
        ring = np.zeros(white.shape, np.uint8)
        cv2.drawContours(ring, [cv2.boxPoints((r[0], (w + 14, h + 14), r[2])).astype(np.int32)], -1, 1, -1)
        cv2.drawContours(ring, [c], -1, 0, -1)
        if dark[ring > 0].mean() < 0.4:
            continue
        m = np.zeros(white.shape, np.uint8)
        cv2.drawContours(m, [c], -1, 1, -1)
        if bluish[m > 0].mean() > 0.3:  # the LCD panel, not a paper label
            continue
        box = cv2.boxPoints(r)
        # refine: the polygon's own corners (minAreaRect is a good start under mild perspective)
        poly = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True).reshape(-1, 2).astype(np.float32)
        if len(poly) == 4:
            box = poly
        out.append((a, box, np.array(r[0])))
    return out


def label_corners(frame):
    q = sorted(_quads(frame), key=lambda t: -t[0])
    if len(q) < 2:
        return None
    (a1, b1, c1), (a2, b2, c2) = q[0], q[1]
    if a1 < 1.2 * a2 or np.linalg.norm(c1 - c2) > 6 * np.sqrt(a1):  # the long label is clearly the bigger one, and near
        return None
    # u: the long label's long side; v: towards the short label (sign fixes the ordering)
    e = [b1[(i + 1) % 4] - b1[i] for i in range(4)]
    u = max(e, key=np.linalg.norm)
    u = u / np.linalg.norm(u)
    v = c2 - c1
    v = v - (v @ u) * u
    v = v / (np.linalg.norm(v) + 1e-9)
    if u[0] * v[1] - u[1] * v[0] < 0:  # every camera looks down on the plate: keep (u, v) right-handed in the image
        u = -u

    def order(b, c):
        d = b - c
        su, sv = d @ u, d @ v
        pick = lambda fu, fv: b[int(np.argmax(fu * su + fv * sv))]
        return np.array([pick(-1, -1), pick(1, -1), pick(1, 1), pick(-1, 1)])

    pts = np.vstack([order(b1, c1), order(b2, c2)]).astype(np.float32)
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pts = cv2.cornerSubPix(g, pts.reshape(-1, 1, 2), (4, 4), (-1, -1),
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)).reshape(-1, 2)
    return pts


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        f = cv2.imread(p)
        c = label_corners(f)
        print(p, None if c is None else np.round(c).astype(int).tolist())
        if c is not None:
            for i, (x, y) in enumerate(c):
                cv2.circle(f, (int(x), int(y)), 5, (0, 0, 255), 2)
                cv2.putText(f, str(i), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(p.replace(".png", "_labels.jpg"), f)
