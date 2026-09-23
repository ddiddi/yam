import sys, cv2, numpy as np, warnings; warnings.simplefilter("ignore"); sys.path.insert(0, '/Users/solotech/yam')
import pyarrow.parquet as pq, pick_place as pp
ep, i = int(sys.argv[1]), int(sys.argv[2])
t = pq.read_table(f"/Users/solotech/yam/datasets/yam_zone_swap/data/chunk-000/episode_{ep:06d}.parquet").to_pydict()
st = np.array(t["observation.state"]); act = np.array(t["action"])
fr = {}
for k in ("cam0", "cam1", "cam2"):
    cap = cv2.VideoCapture(f"/Users/solotech/yam/datasets/yam_zone_swap/videos/chunk-000/observation.images.{k}/episode_{ep:06d}.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read(); fr[k] = f if ok else None
class LC:
    def __init__(s, f): s.f = f
    def latest(s): return s.f
models = pp.load_cameras(); pp.FUSED_MODELS.update(models); pp.FUSED_CAMS.update({k: LC(fr[k]) for k in models if fr.get(k) is not None})
class FakeArm:
    last_cmd = act[i]
    def q(s): return st[i][:6]
args = pp.Args(plan=True, object=0, grasp="top", z_grasp=0.010, max_width=0.092, given_only=True, given="0.2887,-0.1133,0.090,0.015")
pp.MAX_OBJ_WIDTH = 0.092
planner = pp.AzPlanner(); plan = pp.make_plan(pp.with_given([], args), 0, args, planner)
off, conf, note = pp.jaw_alignment(plan, planner, FakeArm(), f"replay_{ep}_{i}")
print("REPLAY:", note, "->", round(off * 100, 1), "cm, confidence", round(conf, 2))
