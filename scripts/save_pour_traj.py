"""Save one squeeze-pour episode as a self-contained trajectory JSON.

    python scripts/save_pour_traj.py EPISODE LOG POUR_CSV OUT.json "what" '{"result": ...}'
"""
import csv
import json
import sys

import numpy as np
import pyarrow.parquet as pq

ep, log, pour_csv, out, what = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
extra = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
D = "datasets/yam_squeeze_pour"
t = pq.read_table(f"{D}/data/chunk-000/episode_{ep:06d}.parquet").to_pydict()
ts, A, S = np.array(t["timestamp"]), np.array(t["action"]), np.array(t["observation.state"])
idx = [int(np.argmin(abs(ts - x))) for x in np.arange(0, ts[-1], 0.1)]
plan = [json.loads(l) for l in open(f"{D}/meta/episode_plans.jsonl") if json.loads(l)["episode_index"] == ep][0]["plan"]
pour = list(csv.reader(open(pour_csv)))
tac = list(csv.reader(open(f"{D}/meta/tactile/episode_{ep:06d}.csv")))
hdr, rows = tac[0], tac[1:]
step = max(1, len(rows) // int(ts[-1] * 10 + 1))
keep = hdr.index("skin_0") + 1
cmd = next((l.strip() for l in open(log) if l.startswith("command:")), None)
doc = {"what": what, "date": "2026-09-24", **extra,
       "source": {"dataset": D, "episode_index": ep, "log": log, "pour_csv": pour_csv,
                  "tactile_csv": f"{D}/meta/tactile/episode_{ep:06d}.csv"},
       "plan": plan,
       "trajectory_10hz": {"fields": "t, q1..q6 (rad), gripper (0 closed..1 open); action = commanded, state = measured",
                           "t": [round(float(ts[i]), 2) for i in idx],
                           "action": [[round(float(v), 4) for v in A[i]] for i in idx],
                           "state": [[round(float(v), 4) for v in S[i]] for i in idx]},
       "squeeze_log": {"fields": pour[0], "rows": pour[1:]},
       "tactile_10hz": {"fields": hdr[:keep], "rows": [r[:keep] for r in rows[::step]]}}
json.dump(doc, open(out, "w"))
print(out, len(idx), "samples")
