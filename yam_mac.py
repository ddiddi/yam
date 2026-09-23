"""macOS support for the I2RT YAM SDK.

`i2rt` hardcodes python-can's `socketcan` backend, which only exists on Linux.
The YAM's USB-CAN adapter runs candleLight firmware (gs_usb protocol), which
python-can can drive directly over libusb on macOS.  Importing this module
patches `can.interface.Bus` so any request for `socketcan` is transparently
served by a `gs_usb` bus instead.  Everything else in i2rt is untouched.

    import yam_mac                       # must come before i2rt imports
    from i2rt.robots.get_robot import get_yam_robot
    robot = get_yam_robot(channel="can0")  # channel name is ignored on mac
"""

from __future__ import annotations

import sys
import time

import can

CANDLELIGHT_VID = 0x1D50
CANDLELIGHT_PID = 0x606F
STM32_DFU_VID = 0x0483
STM32_DFU_PID = 0xDF11


def find_adapter() -> tuple[int | None, str]:
    """Return (usb_bus_index, human_status) for the first candleLight adapter."""
    import usb.core

    dev = usb.core.find(idVendor=CANDLELIGHT_VID, idProduct=CANDLELIGHT_PID)
    if dev is not None:
        return 0, f"candleLight adapter found (bus {dev.bus} addr {dev.address})"
    if usb.core.find(idVendor=STM32_DFU_VID, idProduct=STM32_DFU_PID) is not None:
        return None, (
            "USB-CAN adapter is in STM32 DFU bootloader mode (0483:df11), not running CAN firmware. "
            "Unplug/replug it without holding the boot button; if it stays in DFU, re-flash candleLight "
            "via https://canable.io/updater/"
        )
    return None, "no candleLight USB-CAN adapter (1d50:606f) found on USB"


if sys.platform == "darwin":
    import gs_usb.gs_usb as _gs

    # gs_usb.start() calls libusb detach_kernel_driver(), which is Linux-only and raises
    # EACCES on macOS. Tell it no kernel driver is attached so it skips that step.
    _orig_start = _gs.GsUsb.start
    _orig_read = _gs.GsUsb.read

    # The adapter occasionally delivers a frame that is not the expected 24-byte (timestamped) size;
    # the stock read() then raises struct.error, which i2rt's retry loop does not catch and which
    # kills its control thread mid-motion. Parse a plain 20-byte frame if that is what arrived,
    # and treat any other odd-sized read as "nothing received".
    def _safe_read(self, frame, timeout_ms):  # noqa: ANN
        hw_ts = (self.device_flags & _gs.GS_CAN_MODE_HW_TIMESTAMP) == _gs.GS_CAN_MODE_HW_TIMESTAMP
        try:
            data = self.gs_usb.read(0x81, frame.__sizeof__(hw_ts), timeout_ms)
        except _gs.usb.core.USBError:
            return False
        n = len(data)
        if n == _gs.GS_USB_FRAME_SIZE_HW_TIMESTAMP:
            _gs.GsUsbFrame.unpack_into(frame, data, True)
        elif n == _gs.GS_USB_FRAME_SIZE:
            _gs.GsUsbFrame.unpack_into(frame, data, False)
        else:
            return False
        return True

    _gs.GsUsb.read = _safe_read

    _state = {"reset_done": False}

    def _mac_start(self, *a, **kw):  # noqa: ANN
        """start() with the USB port reset only on the first open in this process.

        libusb_reset_device makes macOS re-enumerate the adapter, and i2rt opens the bus three times
        during bring-up; resetting every time races the re-enumeration (ENODEV). Later opens follow
        our own clean shutdown() and only need the protocol-level mode reset that start() also sends.
        """
        self.gs_usb.is_kernel_driver_active = lambda _intf: False
        if _state["reset_done"]:
            self.gs_usb.reset = lambda: None  # no control transfer before the reset: that blocks in-kernel
        result = _orig_start(self, *a, **kw)
        _state["reset_done"] = True
        return result

    _gs.GsUsb.start = _mac_start

    from can.interfaces.gs_usb import GsUsbBus as _GsUsbBus

    class _NoEchoGsUsbBus(_GsUsbBus):
        """gs_usb hands back our own transmitted frames (is_rx=False). socketcan never does, and
        i2rt relies on that (DM register replies share the 0x7FF request ID), so drop echoes here."""

        def _recv_internal(self, timeout):  # noqa: ANN
            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                msg, filtered = super()._recv_internal(remaining)
                if msg is None or msg.is_rx:
                    return msg, filtered
                if deadline is not None and time.monotonic() >= deadline:
                    return None, filtered

        def shutdown(self) -> None:
            super().shutdown()
            # gs_usb never releases the claimed interface; without the USB reset on the next open
            # that shows up as "Access denied" on claim_interface. Release it explicitly.
            import usb.util

            try:
                usb.util.dispose_resources(self.gs_usb.gs_usb)
            except Exception:  # noqa: BLE001
                pass

    _orig_bus = can.interface.Bus

    class _MacBus(can.BusABC):  # type: ignore[misc]
        """Thin factory: swap socketcan -> gs_usb, pass everything else through."""

        def __new__(cls, *args, **kwargs):  # noqa: ANN
            bustype = kwargs.pop("bustype", kwargs.pop("interface", None))
            if bustype == "socketcan":
                idx, status = find_adapter()
                if idx is None:
                    raise RuntimeError(status)
                kwargs.pop("channel", None)
                import usb.core

                for attempt in range(8):
                    try:
                        return _NoEchoGsUsbBus(channel="candleLight", index=idx, **kwargs)
                    except (usb.core.USBError, can.CanInitializationError):
                        if attempt == 7:
                            raise
                        time.sleep(1.0)  # adapter re-enumerating after the reset; wait it out
                raise RuntimeError("unreachable")
            return _orig_bus(*args, interface=bustype, **kwargs)

    can.interface.Bus = _MacBus  # type: ignore[assignment]
    can.Bus = _MacBus  # type: ignore[assignment]


# --- driver robustness -------------------------------------------------------------------------
# DMSingleMotorCanInterface.clean_error() sends 3 clear-error frames and never reads the 3 replies.
# Those stale replies accumulate across motor_on() calls and starve the 5-retry response matcher,
# which shows up as "fail to communicate with the motor N" when every motor boots in a fault state
# (e.g. comm-loss 0xD after an aborted run). Drain the bus after clearing so each motor starts clean.
from i2rt.motor_drivers import dm_driver as _dm  # noqa: E402

_orig_clean_error = _dm.DMSingleMotorCanInterface.clean_error


def _clean_error_and_drain(self, motor_id: int) -> None:  # noqa: ANN
    _orig_clean_error(self, motor_id)
    self._drain_bus(timeout_s=0.05)


_dm.DMSingleMotorCanInterface.clean_error = _clean_error_and_drain

# DMChainCanInterface.close() flips `running` and shuts the bus in the same breath, while the control
# thread may be mid-transaction; over libusb that surfaces as a spurious "fail to communicate" at exit.
# Give the loop a few cycles to observe `running=False` before pulling the bus.
_orig_chain_close = _dm.DMChainCanInterface.close


def _chain_close_gracefully(self) -> None:  # noqa: ANN
    self.running = False
    time.sleep(0.1)
    _orig_chain_close(self)


_dm.DMChainCanInterface.close = _chain_close_gracefully
