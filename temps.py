"""Read YAM motor temperatures / fault codes without enabling any motor.

    python temps.py            # clear faults, sample feedback for 5 s, print temps
    python temps.py --seconds 20

Nothing is torqued: each motor only gets a clear-error frame and zero MIT frames
(kp = kd = tau = 0), which a disabled DM motor answers with its state frame.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import tyro

import yam_mac  # noqa: F401  -- must be imported before i2rt
from i2rt.motor_drivers.dm_driver import DMSingleMotorCanInterface, MotorType, ReceiveMode

MOTORS = [(i, MotorType.DM4310) for i in range(1, 8)]  # 6 arm joints + gripper


@dataclass
class Args:
    seconds: float = 5.0
    """How long to sample."""
    hz: float = 20.0


def main(args: Args) -> None:
    logging.basicConfig(level=logging.WARNING)
    print(f"adapter: {yam_mac.find_adapter()[1]}")
    iface = DMSingleMotorCanInterface(channel="can0", bustype="socketcan", receive_mode=ReceiveMode.p16, name="temps")
    zero = bytearray(8)  # pos=vel=kp=kd=tau=0 -> a "do nothing" MIT frame
    try:
        iface._drain_bus()
        for mid, _ in MOTORS:
            iface.clean_error(mid)
        time.sleep(0.2)
        iface._drain_bus()

        samples: dict[int, list] = {mid: [] for mid, _ in MOTORS}
        t_end = time.monotonic() + args.seconds
        while time.monotonic() < t_end:
            for mid, mtype in MOTORS:
                try:
                    msg = iface._send_message_get_response(mid, mid, zero)
                    fb = iface.parse_recv_message(msg, mtype, ignore_error=True)
                    samples[mid].append((fb.error_code, fb.error_message, fb.temperature_mos, fb.temperature_rotor, fb.position))
                except Exception as e:  # noqa: BLE001
                    samples[mid].append(("--", f"no reply: {e}", np.nan, np.nan, np.nan))
            time.sleep(1.0 / args.hz)

        print(f"\n{'motor':>5} {'state':>12} {'mos C':>10} {'rotor C':>12} {'pos rad':>8}  n")
        for mid, _ in MOTORS:
            s = samples[mid]
            mos = np.array([x[2] for x in s], float)
            rot = np.array([x[3] for x in s], float)
            pos = np.array([x[4] for x in s], float)
            last = s[-1]
            print(
                f"{mid:>5} {last[1]:>12} {np.nanmin(mos):4.0f}..{np.nanmax(mos):<4.0f} "
                f"{np.nanmin(rot):5.0f}..{np.nanmax(rot):<5.0f} {np.nanmean(pos):8.2f}  {len(s)}"
            )
    finally:
        iface.close()
        print("closed (no motor was enabled)")


if __name__ == "__main__":
    main(tyro.cli(Args))
