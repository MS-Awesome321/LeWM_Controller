import os, sys, time, clr
from System import Decimal
from System.Globalization import CultureInfo

KINESIS_DIR = r"C:\Program Files\Thorlabs\Kinesis"
os.add_dll_directory(KINESIS_DIR)
sys.path.append(KINESIS_DIR)

clr.AddReference(os.path.join(KINESIS_DIR, "Thorlabs.MotionControl.DeviceManagerCLI.dll"))
clr.AddReference(os.path.join(KINESIS_DIR, "Thorlabs.MotionControl.GenericMotorCLI.dll"))
clr.AddReference(os.path.join(KINESIS_DIR, "Thorlabs.MotionControl.KCube.DCServoCLI.dll"))

from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
from Thorlabs.MotionControl.KCube.DCServoCLI import KCubeDCServo
from Thorlabs.MotionControl.GenericMotorCLI import MotorDirection


def dec(x: float) -> Decimal:
    return Decimal.Parse(str(x), CultureInfo.InvariantCulture)


class KDC101:
    def __init__(
        self,
        serial: str,
        polling_ms: int = 200,
        pos_left_mm: float = 0.0,
        deg_left: float = -8.6,
        pos_right_mm: float = 12.0,
        deg_right: float = 6.9,
    ):
        self.serial = serial
        self.polling_ms = polling_ms
        self.dev = None

        # linear calibration
        self.pos_left_mm = pos_left_mm
        self.deg_left = deg_left
        self.pos_right_mm = pos_right_mm
        self.deg_right = deg_right

        self.deg_per_mm = (deg_right - deg_left) / (pos_right_mm - pos_left_mm)

    # ---- conversion helpers ----
    def pos_to_deg(self, pos_mm: float) -> float:
        return self.deg_left + self.deg_per_mm * (pos_mm - self.pos_left_mm)

    def deg_to_pos(self, deg: float) -> float:
        return self.pos_left_mm + (deg - self.deg_left) / self.deg_per_mm

    # ---- device lifecycle ----
    def connect(self):
        DeviceManagerCLI.BuildDeviceList()

        d = KCubeDCServo.CreateKCubeDCServo(self.serial)
        d.Connect(self.serial)

        if not d.IsSettingsInitialized():
            d.WaitForSettingsInitialized(10000)

        try:
            d.LoadMotorConfiguration(self.serial)
        except Exception:
            pass

        d.StartPolling(self.polling_ms)
        time.sleep(0.1)
        d.EnableDevice()
        time.sleep(0.1)

        self.dev = d

    def disconnect(self):
        if self.dev is None:
            return
        try:
            self.dev.StopPolling()
        except Exception:
            pass
        try:
            self.dev.Disconnect(True)
        except Exception:
            pass
        self.dev = None

    def _wait_stop(self, timeout_s: float):
        t0 = time.time()
        while self.dev.Status.IsMoving:
            time.sleep(0.05)
            if time.time() - t0 > timeout_s:
                raise TimeoutError("KDC101 move timeout")

    # ---- reads ----
    def position_mm(self) -> float:
        return float(str(self.dev.Position))

    def angle(self) -> float:
        return self.pos_to_deg(self.position_mm())

    # ---- motion (all in degrees) ----
    def home(self, timeout_ms: int = 60000):
        self.dev.Home(int(timeout_ms))
        self._wait_stop(timeout_s=timeout_ms / 1000)
        self.move_to(0, timeout_ms=timeout_ms)

    def move_to(self, deg: float, timeout_ms: int = 60000):
        self._check_deg(deg)
        pos = self.deg_to_pos(deg)
        self.dev.MoveTo(dec(pos), int(timeout_ms))
        self._wait_stop(timeout_s=timeout_ms / 1000)

    def move_by(self, delta_deg: float, timeout_ms: int = 60000):
        self.move_to(self.angle() + delta_deg, timeout_ms=timeout_ms)

    def jog(self, step_deg: float, timeout_ms: int = 60000):
        self._check_deg(self.angle() + step_deg)

        direction = MotorDirection.Forward if step_deg > 0 else MotorDirection.Backward
        step_mm = abs(step_deg) / abs(self.deg_per_mm)

        self.dev.SetJogStepSize(dec(step_mm))
        self.dev.MoveJog(direction, int(timeout_ms))
        self._wait_stop(timeout_s=timeout_ms / 1000)

    def _check_deg(self, target_deg: float):
        if not (self.deg_left <= target_deg <= self.deg_right):
            raise ValueError(
                f"KDC101 target {target_deg} deg out of range "
                f"[{self.deg_left}, {self.deg_right}]"
            )


class KDC101Controller(KDC101):
    """
    Controller-style wrapper to match common.hardware naming conventions.
    Accepts port= as alias for serial=.
    """

    def __init__(
        self,
        port: str | None = None,
        *,
        serial: str | None = None,
        polling_ms: int = 200,
        pos_left_mm: float = 0.0,
        deg_left: float = -8.6,
        pos_right_mm: float = 12.0,
        deg_right: float = 6.9,
    ):
        serial_value = serial or port
        if serial_value is None:
            raise ValueError("KDC101Controller requires a serial number (serial= or port=).")
        super().__init__(
            serial=serial_value,
            polling_ms=polling_ms,
            pos_left_mm=pos_left_mm,
            deg_left=deg_left,
            pos_right_mm=pos_right_mm,
            deg_right=deg_right,
        )


if __name__ == "__main__":
    goni = KDC101("27271252")
    goni.connect()

    print("pos(mm):", goni.position_mm(), "angle(deg):", goni.angle())

    goni.home()
    print("after home angle(deg):", goni.angle())

    goni.move_to(1.0)
    print("after move_to(0deg):", goni.angle(), "pos(mm):", goni.position_mm())

    goni.disconnect()
