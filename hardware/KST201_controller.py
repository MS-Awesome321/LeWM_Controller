import os, sys, time, clr
from System import Decimal
from System.Globalization import CultureInfo

KINESIS_DIR = r"C:\Program Files\Thorlabs\Kinesis"
os.add_dll_directory(KINESIS_DIR)
sys.path.append(KINESIS_DIR)

clr.AddReference(os.path.join(KINESIS_DIR, "Thorlabs.MotionControl.DeviceManagerCLI.dll"))
clr.AddReference(os.path.join(KINESIS_DIR, "Thorlabs.MotionControl.GenericMotorCLI.dll"))
clr.AddReference(os.path.join(KINESIS_DIR, "Thorlabs.MotionControl.KCube.StepperMotorCLI.dll"))

from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
from Thorlabs.MotionControl.KCube.StepperMotorCLI import KCubeStepper
from Thorlabs.MotionControl.GenericMotorCLI import MotorDirection


def dec(x: float) -> Decimal:
    return Decimal.Parse(str(x), CultureInfo.InvariantCulture)


class KST201:
    def __init__(
        self,
        serial: str,
        polling_ms: int = 200,
        pos_left: float = 0.0,   # optional min limit in real units (mm)
        pos_right: float  = 13.0,  # optional max limit in real units (mm)
        jog_size: float=1.0
    ):
        self.serial = serial
        self.polling_ms = polling_ms
        self.dev = None

        self.pos_left = pos_left
        self.pos_right = pos_right
        self.jog_size = jog_size

    def connect(self):
        DeviceManagerCLI.BuildDeviceList()

        d = KCubeStepper.CreateKCubeStepper(self.serial)
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
        self.dev.SetJogStepSize(dec(self.jog_size))

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

    def position(self) -> float:
        return float(str(self.dev.Position))

    def _wait_stop(self, timeout_s: float):
        if timeout_s == 0:
            return
        t0 = time.time()
        while self.dev.Status.IsMoving:
            time.sleep(0.05)
            if time.time() - t0 > timeout_s:
                raise TimeoutError("KST201 move timeout")

    def _check_pos(self, target: float):
        # keep simple: only check if limits are provided
        if self.pos_left is not None and target < self.pos_left:
            raise ValueError(f"KST201 target {target} out of range (min {self.pos_left})")
        if self.pos_right is not None and target > self.pos_right:
            raise ValueError(f"KST201 target {target} out of range (max {self.pos_right})")

    def home(self, timeout_ms: int = 0):
        if self.dev.Status.IsMoving:
            return
        self.dev.Home(0)
        self._wait_stop(timeout_s=timeout_ms / 1000)

    def move_to(self, pos: float, timeout_ms: int = 0):
        if self.dev.Status.IsMoving:
            return
        self._check_pos(pos)
        self.dev.MoveTo(dec(pos), 0)
        self._wait_stop(timeout_s=timeout_ms / 1000)

    def move_by(self, delta: float, timeout_ms: int = 0) -> None:
        if self.dev.Status.IsMoving:
            return
        target = self.position() + delta
        self.move_to(target, timeout_ms=timeout_ms)

    def jog(self, direction='+', timeout_ms: int = 0):
        if direction == '+':
            d = MotorDirection.Forward
        elif direction == '-':
            d = MotorDirection.Backward
        else:
            raise ValueError("direction should be either + or -")
        if not self.dev.Status.IsMoving:
            self.dev.MoveJog(d, 0)

    def stop(self) -> None:
        """Stop motion immediately."""
        self.dev.StopImmediate()

    def set_speed(self, max_vel: float, accel: float, min_vel: float = 0.0):
        """
        Set velocity profile for MoveTo / MoveRelative.

        Units are the device real units (typically mm/s and mm/s^2 for ZST stages).
        """
        vp = self.dev.GetVelocityParams()
        vp.MinVelocity = dec(min_vel)
        vp.MaxVelocity = dec(max_vel)
        vp.Acceleration = dec(accel)
        self.dev.SetVelocityParams(vp)

    def get_speed(self):
        """
        Read current velocity profile for MoveTo / MoveRelative.

        Returns
        -------
        dict with keys: min_vel, max_vel, accel
        """
        vp = self.dev.GetVelocityParams()
        return {
            "min_vel": float(str(vp.MinVelocity)),
            "max_vel": float(str(vp.MaxVelocity)),
            "accel": float(str(vp.Acceleration)),
        }


class KST201Controller(KST201):
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
        pos_left: float = 0.0,
        pos_right: float = 13.0,
    ):
        serial_value = serial or port
        if serial_value is None:
            raise ValueError("KST201Controller requires a serial number (serial= or port=).")
        super().__init__(
            serial=serial_value,
            polling_ms=polling_ms,
            pos_left=pos_left,
            pos_right=pos_right,
        )


if __name__ == "__main__":
    X = KST201('26007081')
    X.connect()
    print(X.position())
    X.dev.MoveJog(MotorDirection.Forward, 0)
    print(X.position())
    time.sleep(2)
    print(X.dev.Status.IsMoving)
    time.sleep(2)
    print(X.position())
    X.stop()
    time.sleep(1)
    print(X.dev.Status.IsMoving)
    X.disconnect()
