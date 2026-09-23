"""Make the YAM follower arm dance.

Usage (from the yam/ folder, venv activated):

    python dance.py                 # real arm over the USB-CAN adapter
    python dance.py --sim           # MuJoCo only, no hardware needed
    python dance.py --duration 60 --speed 1.3 --amplitude 0.8
    python dance.py --gripper no_gripper

Press Ctrl-C at any time: the arm glides back to rest and torques off.

Choreography: the arm rises from its folded rest pose to a "ready" pose, then
cycles through a set of moves (sway, wave, twist, bounce, gripper snaps ...)
that are all smooth sinusoids layered on top of the ready pose, so joint
speeds stay gentle and everything stays well inside the joint limits.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import dataclass

import numpy as np
import tyro

import yam_mac  # noqa: F401  -- must be imported before i2rt: routes socketcan -> gs_usb on macOS
from i2rt.motor_config_tool.dm_motor_registers import DMRegAddr, read_register, write_register
from i2rt.motor_config_tool.utils import RawCanInterface
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType

# Joint order: [base_yaw, shoulder, elbow, wrist_pitch, wrist_roll, wrist_yaw, (gripper 0=closed..1=open)]
REST = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
READY = np.array([0.0, 0.7, 1.0, 0.0, 0.0, 0.0])

# Conservative envelope the choreography is clipped to (rad). Inside the XML limits with margin.
SAFE_LO = np.array([-1.2, 0.15, 0.3, -1.2, -1.2, -1.5])
SAFE_HI = np.array([1.2, 1.6, 2.2, 1.2, 1.2, 1.5])


@dataclass
class Move:
    name: str
    seconds: float
    # amplitude (rad) and frequency multiplier per joint for the sinusoid layered on READY
    amp: np.ndarray
    freq: np.ndarray
    phase: np.ndarray
    gripper: str = "hold"  # "hold" | "snap" | "breathe"


def _m(name: str, seconds: float, amp: list, freq: list, phase: list | None = None, gripper: str = "hold") -> Move:
    return Move(
        name,
        seconds,
        np.array(amp, dtype=float),
        np.array(freq, dtype=float),
        np.zeros(6) if phase is None else np.array(phase, dtype=float),
        gripper,
    )


CHOREO: list[Move] = [
    _m("sway", 6, amp=[0.6, 0, 0, 0, 0, 0], freq=[1, 0, 0, 0, 0, 0], gripper="breathe"),
    _m("wave", 6, amp=[0, 0, 0, 0.7, 0, 0], freq=[0, 0, 0, 2, 0, 0], gripper="snap"),
    _m("twist", 6, amp=[0, 0, 0, 0, 0.9, 0.9], freq=[0, 0, 0, 0, 1, 1], phase=[0, 0, 0, 0, 0, math.pi]),
    _m("bounce", 6, amp=[0, 0.35, 0.5, 0, 0, 0], freq=[0, 2, 2, 0, 0, 0], phase=[0, 0, math.pi, 0, 0, 0], gripper="snap"),
    _m("figure-8", 8, amp=[0.7, 0.3, 0, 0.5, 0, 0], freq=[1, 2, 0, 2, 0, 0], phase=[0, math.pi / 2, 0, 0, 0, 0]),
    _m("disco", 8, amp=[0.5, 0.3, 0.4, 0.6, 0.8, 1.0], freq=[1, 2, 2, 3, 1, 2], gripper="breathe"),
]


@dataclass
class Args:
    """Make the YAM arm dance."""

    channel: str = "can0"
    """CAN channel (ignored on macOS, the gs_usb adapter is auto-detected)."""
    arm: str = "yam"
    """Arm variant: yam, yam_pro, yam_ultra, yam_ultra_2, big_yam."""
    gripper: str = "linear_4310"
    """Gripper: linear_4310, linear_3507, crank_4310, flexible_4310, no_gripper."""
    sim: bool = False
    """Run in MuJoCo simulation only (no hardware)."""
    duration: float = 0.0
    """Total dance time in seconds; 0 = play the choreography once."""
    speed: float = 0.5
    """Tempo multiplier (0.5 = slow default, 1.0 = brisk, 1.5 = fast)."""
    amplitude: float = 1.0
    """Scale of every move (1.0 = normal, 0.5 = half range)."""
    hz: float = 100.0
    """Command rate."""
    check_only: bool = False
    """Just verify the choreography stays inside the safe envelope and exit."""
    bad_rotor_sensor: tuple[int, ...] = ()
    """Motor ids whose coil NTC reads garbage (e.g. 5). Their firmware over-temp trip is raised in RAM
    (not Flash: reverts on power cycle) and the temperature guards use their MOSFET sensor instead."""
    ot_override: float = 160.0
    """OT_Value (C) written to the motors in --bad-rotor-sensor."""


def pose_at(move: Move, t: float, amplitude: float, tempo: float, ramp: float) -> tuple[np.ndarray, float]:
    """Arm joints + gripper for `t` seconds into `move`. `ramp` in [0,1] fades the amplitude in/out."""
    w = 2 * math.pi * tempo * t
    arm = READY + ramp * amplitude * move.amp * np.sin(move.freq * w + move.phase)
    arm = np.clip(arm, SAFE_LO, SAFE_HI)
    if move.gripper == "snap":
        grip = 1.0 if math.sin(2 * w) > 0 else 0.15
    elif move.gripper == "breathe":
        grip = 0.55 + 0.45 * math.sin(w)
    else:
        grip = 1.0
    return arm, float(np.clip(grip, 0.0, 1.0))


class ControlLoopDead(RuntimeError):
    """The robot's background control thread has exited; commands are no longer reaching the motors."""


def alive(robot) -> bool:  # noqa: ANN
    chain = getattr(robot, "motor_chain", None)
    return chain is None or getattr(chain, "running", True)


T_ROTOR_START_MAX = 60.0  # refuse to start if any rotor is hotter than this (C)
T_ROTOR_ABORT = 80.0  # stop the dance if any rotor crosses this (C); DM motors cut out ~110


class MotorTooHot(RuntimeError):
    pass


def rotor_temps(robot, bad_sensor: tuple[int, ...] = ()) -> np.ndarray | None:  # noqa: ANN
    """Per-motor rotor temperature; motors in `bad_sensor` report their MOSFET temperature instead."""
    state = getattr(robot, "_joint_state", None)
    t = getattr(state, "temp_rotor", None)
    if t is None:
        return None
    t = np.asarray(t, dtype=float).copy()
    mos = getattr(state, "temp_mos", None)
    if mos is not None:
        for mid in bad_sensor:
            if 1 <= mid <= len(t):
                t[mid - 1] = float(mos[mid - 1])
    return t


def check_temps(robot, limit: float, bad_sensor: tuple[int, ...] = ()) -> None:  # noqa: ANN
    t = rotor_temps(robot, bad_sensor)
    if t is not None and t.max() > limit:
        raise MotorTooHot(f"motor {int(t.argmax()) + 1} rotor at {t.max():.0f} C (limit {limit:.0f} C)")


def raise_ot_threshold(channel: str, motor_ids: tuple[int, ...], value: float) -> None:
    """Write OT_Value in RAM (0x55 write, never the 0xAA save) so a motor with a broken coil NTC can enable.

    Must run before the robot takes the bus: register access needs an idle bus.
    """
    iface = RawCanInterface(channel=channel, bustype="socketcan", name="dance_ot")
    try:
        for mid in motor_ids:
            before = read_register(iface, mid, DMRegAddr.OT_VALUE)
            after = write_register(iface, mid, DMRegAddr.OT_VALUE, value)
            print(f"motor {mid}: OT_Value {before:.0f} -> {after:.0f} C (RAM only, reverts on power cycle)")
            if abs(after - value) > 0.5:
                raise RuntimeError(f"motor {mid} echoed OT_Value {after}, wanted {value}")
    finally:
        iface.close()


def glide(robot, target: np.ndarray, seconds: float, hz: float = 100.0) -> None:
    """Linearly interpolate from the current pose to `target` (works on real and sim robots)."""
    start = np.array(robot.get_joint_pos(), dtype=float)
    steps = max(1, int(seconds * hz))
    for i in range(steps + 1):
        if not alive(robot):
            raise ControlLoopDead("control loop died during glide")
        a = i / steps
        robot.command_joint_pos((1 - a) * start + a * target)
        time.sleep(seconds / steps)


def check_choreo(amplitude: float) -> None:
    for move in CHOREO:
        for t in np.linspace(0, move.seconds, 400):
            raw = READY + amplitude * move.amp * np.sin(move.freq * 2 * math.pi * t + move.phase)
            if np.any(raw < SAFE_LO) or np.any(raw > SAFE_HI):
                print(f"  {move.name:10s} clipped at t={t:.2f}: {np.round(raw, 2)}")
    print("choreography check done")


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.check_only:
        check_choreo(args.amplitude)
        return

    arm_type = ArmType.from_string_name(args.arm)
    gripper_type = GripperType.from_string_name(args.gripper)
    has_gripper = gripper_type != GripperType.NO_GRIPPER

    if not args.sim:
        _, status = yam_mac.find_adapter() if sys.platform == "darwin" else (0, "linux socketcan")
        print(f"adapter: {status}")
        if args.bad_rotor_sensor:
            raise_ot_threshold(args.channel, args.bad_rotor_sensor, args.ot_override)

    robot = get_yam_robot(channel=args.channel, arm_type=arm_type, gripper_type=gripper_type, sim=args.sim)
    n = robot.num_dofs()

    def full(arm: np.ndarray, grip: float) -> np.ndarray:
        return np.concatenate([arm, [grip]]) if has_gripper else arm.copy()

    dt = 1.0 / args.hz
    try:
        print(f"robot has {n} dofs, current pos {np.round(robot.get_joint_pos(), 2)}")
        t = rotor_temps(robot, args.bad_rotor_sensor)
        if t is not None:
            note = f"  (motors {list(args.bad_rotor_sensor)} show MOSFET temp)" if args.bad_rotor_sensor else ""
            print(f"rotor temps (C): {np.round(t).astype(int)}{note}")
        check_temps(robot, T_ROTOR_START_MAX, args.bad_rotor_sensor)
        print("rising to READY ...")
        glide(robot, full(READY, 1.0), 3.0, args.hz)
        time.sleep(0.5)

        t_start = time.monotonic()
        loop = 0
        while True:
            for move in CHOREO:
                dur = move.seconds / args.speed
                print(f"[{loop}] {move.name}  ({dur:.1f}s)")
                m0 = time.monotonic()
                while True:
                    t = time.monotonic() - m0
                    if t >= dur:
                        break
                    if args.duration and (time.monotonic() - t_start) >= args.duration:
                        raise KeyboardInterrupt
                    # 1 s fade-in / fade-out so moves blend without a jerk at the seam
                    ramp = min(1.0, t / 1.0, (dur - t) / 1.0)
                    arm, grip = pose_at(move, t, args.amplitude, args.speed, ramp)
                    if not alive(robot):
                        raise ControlLoopDead(f"control loop died during '{move.name}'")
                    if int(t * args.hz) % int(2 * args.hz) == 0:  # every ~2 s
                        check_temps(robot, T_ROTOR_ABORT, args.bad_rotor_sensor)
                    robot.command_joint_pos(full(arm, grip))
                    time.sleep(dt)
            loop += 1
            if args.duration == 0:
                break
    except KeyboardInterrupt:
        print("\nstopping ...")
    except ControlLoopDead as e:
        print(f"\nABORT: {e}. Motors fall back to damping mode; not sending further commands.")
    except MotorTooHot as e:
        print(f"\nTOO HOT: {e}. Returning to rest to let it cool.")
    finally:
        try:
            if alive(robot):
                print("returning to READY, then REST")
                glide(robot, full(READY, 1.0), 2.0, args.hz)
                glide(robot, full(REST, 1.0), 3.0, args.hz)
        except ControlLoopDead as e:
            print(f"ABORT during return: {e}")
        finally:
            robot.close()
        print("done")


if __name__ == "__main__":
    main(tyro.cli(Args))
