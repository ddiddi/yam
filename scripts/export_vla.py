"""Merge every recorded LeRobot dataset under datasets/ into one VLA training set (LeRobot v2.1):

    python scripts/export_vla.py [--out datasets/yam_vla] [--all]

- episodes: only runs marked completed in their dataset's meta/episode_plans.jsonl (--all: every episode, with the
  completion flag in the manifest); datasets without plans are listed in the manifest and left out unless --all
- cameras: two consistent views - observation.images.main = cam1 (the detection camera in every setup) and
  observation.images.aux = cam0; any other camera is dropped. Episodes missing either video are left out.
- the language instruction is each episode's recorded task; tasks are re-indexed across datasets
- videos are hard-linked (no extra disk), parquet re-indexed (episode_index, index, task_index)
- meta/sources.jsonl: for every exported episode, its source dataset, episode and plan (object, grasp, place)
Episodes still being written (not yet in the source's meta/episodes.jsonl) are skipped: re-run to add them.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1] / "datasets"
OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else ROOT / "yam_vla"
ALL = "--all" in sys.argv
CAMS = {"observation.images.cam1": "observation.images.main", "observation.images.cam0": "observation.images.aux"}


def jl(p):
    return [json.loads(l) for l in open(p)] if p.exists() else []


if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "meta").mkdir(parents=True)
tasks: dict[str, int] = {}
episodes, stats, sources, skipped = [], [], [], []
features = None
gidx = 0  # global frame index
ep_out = 0
for src in sorted(p for p in ROOT.iterdir() if p.is_dir() and p != OUT and (p / "meta" / "info.json").exists()):
    info = json.loads((src / "meta" / "info.json").read_text())
    if not all(k in info["features"] for k in CAMS):
        skipped.append({"dataset": src.name, "why": f"lacks {[k for k in CAMS if k not in info['features']]}"})
        continue
    plans = {p["episode_index"]: p for p in jl(src / "meta" / "episode_plans.jsonl")}
    st = {s["episode_index"]: s["stats"] for s in jl(src / "meta" / "episodes_stats.jsonl")}
    if features is None:
        features = {k: v for k, v in info["features"].items() if not k.startswith("observation.images.")}
        for k, new in CAMS.items():
            features[new] = info["features"][k]
    for ep in jl(src / "meta" / "episodes.jsonl"):
        i = ep["episode_index"]
        plan = plans.get(i)
        ok = plan.get("completed") if plan else None
        row = {"dataset": src.name, "episode": i, "task": ep["tasks"][0], "completed": ok, "frames": ep["length"]}
        chunk = i // info["chunks_size"]
        pq_in = src / info["data_path"].format(episode_chunk=chunk, episode_index=i)
        vids = {k: src / info["video_path"].format(episode_chunk=chunk, video_key=k, episode_index=i) for k in CAMS}
        if not ALL and ok is not True:
            skipped.append({**row, "why": "not marked completed" if ok is False else "no success record"})
            continue
        if not pq_in.exists() or not all(v.exists() for v in vids.values()):
            skipped.append({**row, "why": "missing parquet or video"})
            continue
        t = tasks.setdefault(ep["tasks"][0], len(tasks))
        tab = pq.read_table(pq_in)
        n = tab.num_rows
        tab = tab.set_column(tab.schema.get_field_index("episode_index"), "episode_index", pa.array(np.full(n, ep_out, np.int64)))
        tab = tab.set_column(tab.schema.get_field_index("index"), "index", pa.array(np.arange(gidx, gidx + n, dtype=np.int64)))
        tab = tab.set_column(tab.schema.get_field_index("task_index"), "task_index", pa.array(np.full(n, t, np.int64)))
        out_chunk = ep_out // 1000
        dst = OUT / f"data/chunk-{out_chunk:03d}/episode_{ep_out:06d}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(tab, dst)
        for k, new in CAMS.items():
            v = OUT / f"videos/chunk-{out_chunk:03d}/{new}/episode_{ep_out:06d}.mp4"
            v.parent.mkdir(parents=True, exist_ok=True)
            os.link(vids[k], v)
        s = {k: v for k, v in st.get(i, {}).items() if not k.startswith("observation.images.")}
        for k, new in CAMS.items():
            if k in st.get(i, {}):
                s[new] = st[i][k]
        for k, arr in (("episode_index", np.full(n, ep_out)), ("index", np.arange(gidx, gidx + n)), ("task_index", np.full(n, t))):
            s[k] = {"min": [int(arr.min())], "max": [int(arr.max())], "mean": [float(arr.mean())], "std": [float(arr.std())], "count": [n]}
        stats.append({"episode_index": ep_out, "stats": s})
        episodes.append({"episode_index": ep_out, "tasks": [ep["tasks"][0]], "length": n})
        sources.append({"episode_index": ep_out, **row, "plan": (plan or {}).get("plan")})
        gidx += n
        ep_out += 1

info_out = {"codebase_version": "v2.1", "robot_type": "yam", "total_episodes": ep_out, "total_frames": gidx,
            "total_tasks": len(tasks), "total_videos": 2 * ep_out, "total_chunks": ep_out // 1000 + 1, "chunks_size": 1000,
            "fps": 30, "splits": {"train": f"0:{ep_out}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features}
M = OUT / "meta"
(M / "info.json").write_text(json.dumps(info_out, indent=4))
(M / "tasks.jsonl").write_text("".join(json.dumps({"task_index": v, "task": k}) + "\n" for k, v in tasks.items()))
(M / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
(M / "episodes_stats.jsonl").write_text("".join(json.dumps(s) + "\n" for s in stats))
(M / "sources.jsonl").write_text("".join(json.dumps(s) + "\n" for s in sources))
(M / "skipped.jsonl").write_text("".join(json.dumps(s) + "\n" for s in skipped))
print(f"{OUT}: {ep_out} episodes, {gidx} frames ({gidx / 30 / 60:.0f} min), {len(tasks)} tasks; skipped {len(skipped)}")
