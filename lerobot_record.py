"""Record a YAM episode (two cameras + joint state/action) straight into the LeRobot v2.1 dataset layout.

    dataset/
      meta/info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl
      data/chunk-000/episode_000000.parquet
      videos/chunk-000/observation.images.<cam>/episode_000000.mp4   (H.264, yuv420p)

No lerobot / torch / ffmpeg dependency: pyarrow writes the parquet, OpenCV (avc1) writes the videos.
A dataset written here loads with `LeRobotDataset("<path>")` (lerobot >= 0.1, codebase v2.1).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]


class LiveCamera:
    """Continuously grabs frames on a thread so recording never stalls the control loop."""

    def __init__(self, index: int, width: int = 1280, height: int = 720):
        from scene3d import device_index
        self.index = index
        self.cap = cv2.VideoCapture(device_index(index))
        if not self.cap.isOpened():
            raise RuntimeError(f"camera {index} did not open")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame: np.ndarray | None = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        t0 = time.time()
        while self.frame is None and time.time() - t0 < 5.0:
            time.sleep(0.05)
        if self.frame is None:
            raise RuntimeError(f"camera {index} produced no frames")
        time.sleep(1.5)  # exposure settle

    def _loop(self) -> None:
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f
            else:
                time.sleep(0.01)

    def latest(self) -> np.ndarray:
        with self.lock:
            return self.frame.copy()

    def close(self) -> None:
        self.running = False
        self.thread.join(timeout=2.0)
        self.cap.release()


class EpisodeRecorder:
    """Samples (state, action, frames) at `fps` on a background thread between start() and stop()."""

    def __init__(
        self,
        root: Path,
        cameras: dict[str, LiveCamera],
        get_state: Callable[[], np.ndarray],
        get_action: Callable[[], np.ndarray],
        task: str,
        fps: int = 30,
        robot_type: str = "yam",
    ):
        self.root, self.cameras, self.task, self.fps, self.robot_type = Path(root), cameras, task, fps, robot_type
        self.get_state, self.get_action = get_state, get_action
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.stamps: list[float] = []
        self.writers: dict[str, cv2.VideoWriter] = {}
        self.shapes: dict[str, tuple[int, int, int]] = {}
        self.img_sums: dict[str, np.ndarray] = {}
        self.img_sqsums: dict[str, np.ndarray] = {}
        self.img_mins: dict[str, np.ndarray] = {}
        self.img_maxs: dict[str, np.ndarray] = {}
        self.running = False
        self.thread: threading.Thread | None = None
        info_path = self.root / "meta" / "info.json"
        self.prev = json.loads(info_path.read_text()) if info_path.exists() else None  # append to an existing dataset
        self.ep = self.prev["total_episodes"] if self.prev else 0
        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        (self.root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        for cam in cameras:
            (self.root / "videos" / "chunk-000" / f"observation.images.{cam}").mkdir(parents=True, exist_ok=True)

    def _video_path(self, cam: str) -> Path:
        return self.root / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{self.ep:06d}.mp4"

    def start(self) -> None:
        for cam, lc in self.cameras.items():
            h, w = lc.latest().shape[:2]
            self.shapes[cam] = (h, w, 3)
            wr = cv2.VideoWriter(str(self._video_path(cam)), cv2.VideoWriter_fourcc(*"avc1"), self.fps, (w, h))
            if not wr.isOpened():
                raise RuntimeError(f"could not open H.264 writer for {cam}")
            self.writers[cam] = wr
            self.img_sums[cam] = np.zeros(3)
            self.img_sqsums[cam] = np.zeros(3)
            self.img_mins[cam] = np.full(3, np.inf)
            self.img_maxs[cam] = np.full(3, -np.inf)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        period = 1.0 / self.fps
        t0 = time.monotonic()
        i = 0
        while self.running:
            target = t0 + i * period
            now = time.monotonic()
            if now < target:
                time.sleep(target - now)
            state = np.asarray(self.get_state(), dtype=np.float32)
            action = np.asarray(self.get_action(), dtype=np.float32)
            frames = {cam: lc.latest() for cam, lc in self.cameras.items()}
            self.states.append(state)
            self.actions.append(action)
            self.stamps.append(i / self.fps)
            for cam, f in frames.items():
                self.writers[cam].write(f)
                # LeRobot image stats are RGB in [0,1]; every 8th pixel is plenty and keeps the loop at fps
                px = f[::8, ::8, ::-1].reshape(-1, 3).astype(np.float32) / 255.0
                self.img_sums[cam] += px.mean(0)
                self.img_sqsums[cam] += (px**2).mean(0)
                self.img_mins[cam] = np.minimum(self.img_mins[cam], px.min(0))
                self.img_maxs[cam] = np.maximum(self.img_maxs[cam], px.max(0))
            i += 1
        self.wall = time.monotonic() - t0

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        for wr in self.writers.values():
            wr.release()
        n = len(self.states)
        if n and abs(n / self.fps - self.wall) > 0.1 * self.wall:
            print(f"WARNING: recorded {n} frames in {self.wall:.1f}s = {n / self.wall:.1f} fps, not {self.fps}")

    # ------------------------------------------------------------------------------------ dataset files
    def save(self) -> Path:
        n = len(self.states)
        if n == 0:
            raise RuntimeError("no frames recorded")
        states, actions = np.stack(self.states), np.stack(self.actions)
        stamps = np.asarray(self.stamps, dtype=np.float32)
        prev_frames = self.prev["total_frames"] if self.prev else 0
        # tasks: reuse an existing task index or append a new task
        tasks_path = self.root / "meta" / "tasks.jsonl"
        tasks = [json.loads(l) for l in tasks_path.read_text().splitlines() if l.strip()] if tasks_path.exists() else []
        task_index = next((t["task_index"] for t in tasks if t["task"] == self.task), None)
        if task_index is None:
            task_index = len(tasks)
            tasks.append({"task_index": task_index, "task": self.task})
        table = pa.table(
            {
                "observation.state": pa.array(states.tolist(), pa.list_(pa.float32(), 7)),
                "action": pa.array(actions.tolist(), pa.list_(pa.float32(), 7)),
                "timestamp": pa.array(stamps, pa.float32()),
                "frame_index": pa.array(np.arange(n), pa.int64()),
                "episode_index": pa.array(np.full(n, self.ep), pa.int64()),
                "index": pa.array(prev_frames + np.arange(n), pa.int64()),
                "task_index": pa.array(np.full(n, task_index, dtype=np.int64), pa.int64()),
            }
        )
        pq.write_table(table, self.root / "data" / "chunk-000" / f"episode_{self.ep:06d}.parquet")

        features: dict = {
            "observation.state": {"dtype": "float32", "shape": [7], "names": JOINT_NAMES},
            "action": {"dtype": "float32", "shape": [7], "names": JOINT_NAMES},
        }
        for cam, (h, w, c) in self.shapes.items():
            features[f"observation.images.{cam}"] = {
                "dtype": "video",
                "shape": [h, w, c],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": float(self.fps),
                    "video.height": h,
                    "video.width": w,
                    "video.channels": c,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
        for k, dt in (("timestamp", "float32"), ("frame_index", "int64"), ("episode_index", "int64"), ("index", "int64"), ("task_index", "int64")):
            features[k] = {"dtype": dt, "shape": [1], "names": None}

        n_ep = self.ep + 1
        info = {
            "codebase_version": "v2.1",
            "robot_type": self.robot_type,
            "total_episodes": n_ep,
            "total_frames": prev_frames + n,
            "total_tasks": len(tasks),
            "total_videos": len(self.cameras) * n_ep,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": self.fps,
            "splits": {"train": f"0:{n_ep}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
        }
        (self.root / "meta" / "info.json").write_text(json.dumps(info, indent=4))
        tasks_path.write_text("".join(json.dumps(t) + "\n" for t in tasks))
        with (self.root / "meta" / "episodes.jsonl").open("a") as fh:
            fh.write(json.dumps({"episode_index": self.ep, "tasks": [self.task], "length": n}) + "\n")

        def vec_stats(a: np.ndarray) -> dict:
            a = a.reshape(n, -1)
            return {
                "min": a.min(0).tolist(),
                "max": a.max(0).tolist(),
                "mean": a.mean(0).tolist(),
                "std": a.std(0).tolist(),
                "count": [n],
            }

        stats = {
            "observation.state": vec_stats(states),
            "action": vec_stats(actions),
            "timestamp": vec_stats(stamps),
            "frame_index": vec_stats(np.arange(n, dtype=np.float64)),
            "episode_index": vec_stats(np.full(n, self.ep, dtype=np.float64)),
            "index": vec_stats(np.arange(n, dtype=np.float64)),
            "task_index": vec_stats(np.full(n, task_index, dtype=np.float64)),
        }
        for cam in self.cameras:
            mean = self.img_sums[cam] / n
            std = np.sqrt(np.maximum(self.img_sqsums[cam] / n - mean**2, 0.0))
            stats[f"observation.images.{cam}"] = {  # per-channel, shape [3,1,1] as LeRobot expects
                "min": self.img_mins[cam].reshape(3, 1, 1).tolist(),
                "max": self.img_maxs[cam].reshape(3, 1, 1).tolist(),
                "mean": mean.reshape(3, 1, 1).tolist(),
                "std": std.reshape(3, 1, 1).tolist(),
                "count": [n],
            }
        stats["index"] = vec_stats(prev_frames + np.arange(n, dtype=np.float64))
        with (self.root / "meta" / "episodes_stats.jsonl").open("a") as fh:
            fh.write(json.dumps({"episode_index": self.ep, "stats": stats}) + "\n")
        return self.root
