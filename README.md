# yam — I2RT YAM arm on macOS

```
yam/
├── i2rt/            # I2RT SDK (git clone, installed editable into .venv)
├── .venv/           # Python 3.11 venv (uv) with i2rt + gs_usb + pyusb
├── yam_mac.py       # macOS shim: routes i2rt's socketcan calls to gs_usb (libusb)
├── dance.py         # the dance
├── run_dance.sh     # ./run_dance.sh [--sim] [--duration 60] [--speed 1.2] [--amplitude 0.7]
└── README.md
```

## Run

```zsh
./run_dance.sh --sim        # MuJoCo only, no hardware
./run_dance.sh              # real arm (plays the choreography once)
./run_dance.sh --duration 120 --speed 1.3
./run_dance.sh --check-only # verify choreography stays inside the safe envelope
```

Ctrl-C at any time → arm glides back to rest and torques off.

## What `yam_mac.py` fixes (import it before any `i2rt` import)

1. i2rt hardcodes python-can's `socketcan` (Linux only) → redirected to the `gs_usb` backend over libusb.
2. The `gs_usb` package calls a Linux-only `detach_kernel_driver` → skipped on macOS.
3. gs_usb echoes our own transmitted frames back; socketcan never does and i2rt reads them as motor replies
   (DM register reads share the 0x7FF request ID, so every register read came back 0) → echoes are dropped.
4. i2rt's `clean_error()` fires 3 clear-error frames and never reads the 3 replies; when every motor boots in a
   fault state (comm-loss `0xD` after an aborted run) the stale replies starve the 5-retry matcher and bring-up
   dies with "fail to communicate with the motor N" → bus is drained after each clear.
5. `DMChainCanInterface.close()` shuts the bus while the control thread is mid-transaction → waits 100 ms first.

## Hardware notes

* The YAM's USB-CAN adapter runs **candleLight** firmware and should enumerate as USB `1d50:606f`.
  `yam_mac.py` finds it automatically; no `can0` / `ip link` setup is needed on macOS.
* If `system_profiler SPUSBDataType` shows `0483:df11 "DFU in FS Mode"` instead, the adapter is stuck in the
  STM32 bootloader. Unplug/replug it (no button held). If it stays in DFU, re-flash candleLight from Chrome at
  https://canable.io/updater/ (pick your CANable board model → candleLight firmware).
* Motors are zeroed at the folded rest pose; `dance.py` glides from wherever the arm is to READY, and ends at rest.
* Motor LEDs blinking **red** with fault nibble `0xD` = comm-loss timeout (400 ms, factory default). Harmless:
  bring-up clears it. Blinking green = idle, solid green = enabled.
* If nothing on the bus ACKs (adapter reports `CAN_ERR_ACK` error frames) and a power-cycle of the arm supply
  doesn't fix it, reseat the CAN/power connector at the gripper.
* Gripper `linear_4310` auto-calibrates on connect (it will open/close once).
* `i2rt/motor_config_tool/dm_motor_registers.py` opens a raw socketcan socket and dies on macOS with
  `OSError: [Errno 43] Protocol not supported`. Invoke it through the shim instead:
  `python -c "import sys, yam_mac, runpy; sys.argv=['x','read','OT_Value','--motor-id','5','--channel','can0']; runpy.run_path('i2rt/i2rt/motor_config_tool/dm_motor_registers.py', run_name='__main__')"`
* **Joint 5 reports a false over-temperature** (error `0xC`, kills the control loop on the first motion).
  Its rotor sensor reads 95-120 C while its own MOSFET reads 30 C, every other joint sits at ambient, the
  reading does not fall after many minutes unpowered, and it swings 20 C within one 5-second sample — a
  dead rotor NTC. `OT_Value` (DM register 2) is 100.0 C on every joint, which that noise straddles.
  `temps.py` shows all of this without enabling a motor.

## Rebuild the env

```zsh
brew install libusb cmake
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install --build-constraints .build-constraints.txt -e ./i2rt gs_usb pyusb
```
(`.build-constraints.txt` pins `scikit-build-core<0.10`, required to build the `ruckig` dependency.)

## Pick the blue bottle (`pick_bottle.py`)

```zsh
python pick_bottle.py --detect                      # camera only: find the bottle -> captures/detect.jpg
python pick_bottle.py --calibrate                   # ARM MOVES, table must be clear: 6-point image<->table fit -> calib.json
python pick_bottle.py --plan [--grasp side]         # camera only: target xy + IK check, arm untouched
python pick_bottle.py --pick --grasp side --record datasets/yam_pick_bottle   # ARM MOVES; records a LeRobot v2.1 dataset
```

* Cam 0 (C270, low oblique view from the robot's front-right) drives detection; cam 1 (side view) is recorded only.
* Re-run `--calibrate` whenever cam 0 moves. It fits a homography on the z = 9 cm plane between the closed
  fingertip and the bottle's cap-top centre — both must be at that height or the ~30° camera turns any height
  mismatch into ~2x lateral error.
* The real arm's FK is off by a few cm from the camera depending on the IK branch, so the pick blinks the gripper
  8 cm in front of the bottle at the calibration height and corrects the command before grasping (`--no-servo` skips it).
* `--grasp top` descends over the cap; `--grasp side` tilts the wrist 60° and clamps the body from behind (worked first try).
* `lerobot_record.py` writes the dataset (parquet via pyarrow, H.264 mp4 via OpenCV, meta/*.json[l]); no lerobot install needed.

## Pick and place any object (`pick_place.py`)

```zsh
python pick_place.py --background                # camera only, TABLE MUST BE EMPTY: the reference frame
python pick_place.py --survey                    # camera only: every object on the table + its grasp mode
python pick_place.py --plan --object 0           # camera only: the 4-step plan -> plan.json, captures/plan.jpg
python pick_place.py --plan --frames captures/scene_cam0.jpg   # ... offline, from a saved frame
python pick_place.py --run --object 0            # ARM MOVES: look, plan, pick, hold 5 s, place 5 cm away
python pick_place.py --run --no-rescan --record None            # ... without the mid-run stops or the dataset
```

The four steps, all of them checked before anything moves:

1. **Identify** — cam 0 minus `captures/bg_cam0.png` gives one blob per object; its lowest full-width row is
   where it touches the table (ray x z=0 -> footprint centre), the silhouette width there is the diameter and
   the top row is the height. Same extraction as `scene3d.py`, minus its 4 cm height floor, so flat things count.
2. **Trajectory** — every *other* object becomes a no-go cylinder (radius + 2.5 cm, its own height + 1 cm).
   Wrist yaw, approach azimuth and travel height are searched for the largest clearance, then the whole path is
   swept in MuJoCo: every arm/gripper mesh vertex — and, after the grasp, the carried object — must stay outside
   every cylinder and above the table. `--plan` prints the clearance of each leg; a negative one is refused.
3. **Vertical or horizontal** — from flatness (h/d) and height: below 4.5 cm tall, or h/d < 0.6, or above 16 cm
   tall -> **horizontal** (wrist tilted 60°, fingers come in from the side and pinch across the body); otherwise
   **vertical** (straight down the object's axis). If the chosen mode has no collision-free trajectory the other
   is tried and the fallback is printed. `--grasp top|side` forces it.
4. **Pick and place** — the jaw pre-opens to the object's width + 1.2 cm per side, closes on contact (no need to
   know the width), lifts, holds `--hold` (5 s), moves `--place-dist` (5 cm) along the clearest direction,
   sets down, releases, retreats, rests. `--place-dir <deg>` forces the direction.

`--run` needs no other stage first — it looks, plans and only then moves — and it **stops twice more on the way**
to re-read the table: hovering above the object before it descends, and holding the object before it carries it
across. Each stop projects the arm's own meshes into the frame and masks them out before detecting, so the robot
is not read as an object; anything hidden *behind* that mask keeps its last known position, because it is still
there to be knocked over. New, moved and removed objects are printed, the rest of the trajectory is re-validated
from the current pose, and a place point that is no longer free is replaced. If nothing safe is left, the arm sets
the object back down where it picked it up and parks. `--no-rescan` disables the stops, `--pause` sets their length.

Every `--run` writes a LeRobot v2.1 dataset to `datasets/yam_pick_place` (both cameras + state/action at 30 fps),
partial episodes included, so an aborted attempt is still on file for the next iteration. `--record None` turns it off.

* `AzPlanner` overrides `Planner.topdown` as an instance method, which is what lets a side grasp come in from a
  non-radial direction: every inherited helper (`ik`, `move_cartesian`, `settle`, `visual_correct_3d`) follows.
* The visual check from `pick_bottle.py` runs first (`--no-servo` skips it); the corrected xy shifts the pick
  *and* the place point, so the trajectory is re-validated against the obstacles before the arm commits to it.
* Closing on nothing (measured opening < 6 mm) aborts the place and parks the arm instead of miming a drop.
* **`captures/bg_cam0.png` must be from the current session**, captured with the table clear and the arm
  parked (`--background` does both cameras). Everything present in it is invisible afterwards: with a stale
  background a bottle that was there yesterday is never detected, while a cable that has since been moved is.
  The parked arm being *in* the frame is what makes it cancel out; the mid-run rescans mask it explicitly.
* The detection window is the workspace itself, not a pixel rectangle: every point of the reachable ring of
  table, extruded to `MAX_OBJ_HEIGHT` and projected into the image. A blob that reaches the edge of it is cut
  there, so its lowest row is not where it touches the table or its top row is not its top — a cable trailing
  up into the base plate measured 4 x 20 cm and was planned as a tall side grasp until this went in. **The
  object's whole base must be inside the frame**, or it cannot be located at all. Skipped blobs are printed
  with their rough position, so "no objects found" is never silent.
* `table_foreground` suppresses pixels whose **hue** still matches the background, whatever their brightness
  and saturation. The C270 re-meters as soon as anything is placed on the desk: half the tabletop came back
  brighter and washed out (saturation 137 → 16), passed `scene3d.foreground`'s darker-than-background shadow
  test, and merged the chair, the table edge and the bottle into one 394x720 blob. Hue barely moves under a
  light change; on that frame the rule dropped 96% of the false area and kept 84% of the bottle. It also means
  a grey object whose noisy hue lands near the table's is suppressed — a background captured under the run's
  own lighting is the real fix, so the suppressed fraction is printed when it is large.

## 3D map, camera calibration, replay

```zsh
python calib3d.py --sweep [--grid low --xy "0.22,-0.04;0.42,-0.28"]  # ARM MOVES: fingertips blink at a 3D grid, both cams
python calib3d.py --solve      # intrinsics (f, k1; principal point fixed at centre) + pose per camera -> cameras.json
python scene3d.py              # both cameras + arm pose -> scene.json (objects: table-contact ray/plane, cap triangulation)
python build_viewer.py         # scene.json -> scene_viewer.html (three.js; publish as an artifact)
python pick_bottle.py --pick --grasp side --target "x,y" --z-grasp 0.03 --contact-close --record datasets/<name>
python replay.py datasets/<name> --episode 0 --record   # stream the recorded actions back; appends as a new episode
python recover.py [--azimuth 0]  # bring the arm home from any pose (swing, lift, rest)
```

* `captures/bg_cam*.png` must be an empty-table capture from the current session (lighting, cable position).
* Cam 1 (table-level side view) sees the fingers against the robot's black body, so its sweep detections are
  hand-labelled in `captures/sweep_low/cam1_labels.json`; cam 0 is fully automatic (4 px RMS).
* The visual correction keeps the desired check point fixed and accumulates (actual − desired); measuring
  against a moving check point re-applies the arm's constant FK offset every iteration (that bug cost two picks).
* Replay is open-loop joint playback: it re-traces the motion, not the object — if the object moved, it misses.
