"""Live preview of the overhead camera.   python cam_view.py [--cam 0]
q / Esc closes the window; s saves a frame to captures/snap_<n>.jpg.
"""
import sys, time, cv2

cam = int(sys.argv[sys.argv.index("--cam") + 1]) if "--cam" in sys.argv else 0
cap = cv2.VideoCapture(cam)
if not cap.isOpened():
    sys.exit(f"camera {cam} did not open")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
n = 0
win = f"camera {cam}"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
while True:
    ok, frame = cap.read()
    if not ok:
        time.sleep(0.05)
        continue
    cv2.imshow(win, frame)
    k = cv2.waitKey(15) & 0xFF
    if k in (ord("q"), 27):
        break
    if k == ord("s"):
        path = f"captures/snap_{n}.jpg"; cv2.imwrite(path, frame); n += 1; print("saved", path)
cap.release()
cv2.destroyAllWindows()
