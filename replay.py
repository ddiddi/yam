"""Replay a recorded LeRobot episode on the arm: stream its `action` joint targets at the recorded timestamps.

    python replay.py datasets/yam_pick_cups [--episode 0] [--record] [--speed 1.0] [--dry]

--record appends the replay to the same dataset as a new episode (task "<task> (replay)").
The arm first glides from wherever it is to the episode's first action (4 s), then follows the trajectory.
Ctrl-C: the arm glides back to rest and torques off.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import yam_mac  # noqa: F401
from pick_bottle import Arm, Planner, REST, goto_joint
from lerobot_record import EpisodeRecorder, LiveCamera

root = Path(sys.argv[1])
ep = int(sys.argv[sys.argv.index("--episode") + 1]) if "--episode" in sys.argv else 0
speed = float(sys.argv[sys.argv.index("--speed") + 1]) if "--speed" in sys.argv else 1.0
record, dry = "--record" in sys.argv, "--dry" in sys.argv
info = json.loads((root / "meta/info.json").read_text())
t = pq.read_table(root / info["data_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep))
actions = np.array(t["action"].to_pylist(), dtype=np.float64)
stamps = np.array(t["timestamp"].to_pylist(), dtype=np.float64) / speed
task = [json.loads(l) for l in (root / "meta/tasks.jsonl").read_text().splitlines() if l.strip()][int(t["task_index"][0].as_py())]["task"]
print(f"episode {ep}: {len(actions)} actions over {stamps[-1]:.1f} s, task '{task}'")
if dry:
    print("first action", np.round(actions[0], 3), "last", np.round(actions[-1], 3)); sys.exit(0)

planner = Planner()
arm = Arm(False, "can0", 100.0)
rec = None; cams = {}
if record:
    cams = {"cam0": LiveCamera(0), "cam1": LiveCamera(1)}
    rec = EpisodeRecorder(root, cams, arm.state7, lambda: arm.last_cmd, f"{task} (replay)", info["fps"])
try:
    print("gliding to the first action")
    arm.grip = float(actions[0][6])
    goto_joint(arm, planner, actions[0][:6], 4.0)
    if rec:
        rec.start()
    t0 = time.monotonic()
    for a, ts in zip(actions, stamps):
        while time.monotonic() - t0 < ts:
            time.sleep(0.001)
        if not arm.alive():
            raise RuntimeError("control loop died")
        arm.grip = float(a[6])
        arm._cmd(a[:6])
    print("trajectory done; returning to rest")
    if rec:
        rec.stop()
    arm.grip = 1.0
    goto_joint(arm, planner, REST, 5.0)
except KeyboardInterrupt:
    print("interrupted: returning to rest")
    if rec:
        rec.stop()
    arm.grip = 1.0
    goto_joint(arm, planner, REST, 5.0)
finally:
    arm.close()
    for c in cams.values():
        c.close()
if rec:
    rec.save(); print(f"replay saved as episode {rec.ep} of {root}")
