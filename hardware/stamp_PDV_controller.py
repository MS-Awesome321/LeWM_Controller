from __future__ import annotations
from typing import Optional
import time

# Import vendor functions (your existing file)
from .drivers.kz100_pdv_api import *  # noqa: F403

STEPS_PER_MM = 3280.0


def mm_to_steps(mm: float) -> int:
    return int(round(mm * STEPS_PER_MM))


def steps_to_mm(steps: int) -> float:
    return steps / STEPS_PER_MM


class PDVStage:
    def __init__(
        self,
        com_port: str,
        channel: str,                 # 'X','Y','Z','L'
        pos_min_mm: float = 0.0,
        pos_max_mm: float = 195.0,
        max_speed: Optional[int] = None,   # native units 0..16383
    ):
        self.com_port = com_port
        self.channel = channel
        self.pos_min_mm = pos_min_mm
        self.pos_max_mm = pos_max_mm
        self.max_speed = max_speed
        self.connected = False  # logical only (vendor functions open/close each call)

    # ---- lifecycle ----
    def connect(self):
        _ = self.position_mm()  # basic comm check
        self.connected = True
        if self.max_speed is not None:
            self.set_max_speed(self.max_speed)

    def disconnect(self):
        self.connected = False

    # ---- safety ----
    def _check_pos(self, target_mm: float):
        if target_mm < self.pos_min_mm or target_mm > self.pos_max_mm:
            raise ValueError(
                f"PDV target {target_mm} mm out of range "
                f"[{self.pos_min_mm}, {self.pos_max_mm}]"
            )

    # ---- read ----
    def position_steps(self) -> int:
        return GetDisplay(self.com_port, self.channel)  # noqa: F405

    def position_mm(self) -> float:
        return steps_to_mm(self.position_steps())

    def status(self, kind: str) -> str:
        return GetStatus(self.com_port, kind)  # noqa: F405

    # ---- motion (mm public API) ----
    def move_by(self, delta_mm: float, timeout_s: float = 10.0):
        target = self.position_mm() + delta_mm
        try:
            self._check_pos(target)
        except ValueError:
            print("Out of range")
        else:
            SetDistance(self.com_port, self.channel, mm_to_steps(delta_mm))  # noqa: F405
            self.wait_until_stopped(timeout_s=timeout_s)

    def move_to(self, pos_mm: float, timeout_s: float = 10.0):
        try:
            self._check_pos(pos_mm)
        except ValueError:
            print("Out of range")
        else:
            cur = self.position_mm()
            self.move_by(pos_mm - cur, timeout_s=timeout_s)

    # ---- speed / stop ----
    def set_speed(self, speed: int):
        SetSpeed(self.com_port, self.channel, int(speed))  # noqa: F405

    def set_max_speed(self, max_speed: int):
        SetMaxSpeed(self.com_port, self.channel, int(max_speed))  # noqa: F405

    def stop(self):
        Stop(self.com_port)  # noqa: F405

    # ---- motion wait helpers ----
    def wait_until_stopped(
        self,
        timeout_s: float = 60.0,
        poll_s: float = 0.05,
        stable_reads: int = 3,
        deadband_steps: int = 5,  # set to 1 if you see +/-1 step jitter
    ):
        """
        Wait until position stops changing (stable for stable_reads polls).
        """
        t0 = time.time()
        last = self.position_steps()
        same = 0

        while True:
            time.sleep(poll_s)
            cur = self.position_steps()

            if abs(cur - last) <= deadband_steps:
                same += 1
                if same >= stable_reads:
                    return
            else:
                same = 0
                last = cur
            #
            # if time.time() - t0 > timeout_s:
            #     raise TimeoutError("PDV move timeout (position did not settle)")


class PDVController(PDVStage):
    """
    Controller-style wrapper to match common.hardware naming conventions.
    Accepts port= as alias for com_port=.
    """

    def __init__(
        self,
        port: str | None = None,
        *,
        com_port: str | None = None,
        channel: str = "L",
        pos_min_mm: float = 0.0,
        pos_max_mm: float = 195.0,
        max_speed: Optional[int] = None,
    ):
        port_value = com_port or port
        if port_value is None:
            raise ValueError("PDVController requires a COM port (com_port= or port=).")
        super().__init__(
            com_port=port_value,
            channel=channel,
            pos_min_mm=pos_min_mm,
            pos_max_mm=pos_max_mm,
            max_speed=max_speed,
        )
