from __future__ import annotations

from typing import Iterable, Optional, Union

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

from .KST201_controller import KST201
from .KDC101_controller import KDC101
from .stamp_PDV_controller import PDVStage


AxisNames = Union[str, Iterable[str]]


class TransferControl:
    """
    Transfer stage stack (x, y, z, goni, L) built from the source-of-truth
    KST201/KDC101/PDV wrappers.
    """

    def __init__(
        self,
        *,
        x_serial: str = "26007081",
        y_serial: str = "26007120",
        z_serial: str = "26006985",
        goni_serial: str = "27271252",
        polling_ms: int = 200,
        L_com_port: str = "COM19",
        L_pos_min_mm: float = 0.0,
        L_pos_max_mm: float = 195.0,
        L_max_speed: Optional[int] = None,
        auto_connect: bool = True,
    ) -> None:
        self._config = {
            "x_serial": x_serial,
            "y_serial": y_serial,
            "z_serial": z_serial,
            "goni_serial": goni_serial,
            "polling_ms": polling_ms,
            "L_com_port": L_com_port,
            "L_pos_min_mm": L_pos_min_mm,
            "L_pos_max_mm": L_pos_max_mm,
            "L_max_speed": L_max_speed,
        }
        self.x = None
        self.y = None
        self.z = None
        self.goni = None
        self.L = None
        self._axes: dict[str, object] = {}
        self._connected = False

        if auto_connect:
            self.connect()

    def connect(self) -> None:
        if self._connected:
            return
        self._init_hardware()
        DeviceManagerCLI.BuildDeviceList()
        visible = list(DeviceManagerCLI.GetDeviceList())
        print("Kinesis sees:", sorted(visible))

        for name in ["x", "y", "z", "goni", "L"]:
            axis = self._axes[name]
            print(f"Connecting {name} ({axis.__class__.__name__}) ...")
            axis.connect()

        self._connected = True
        print("All devices connected.")

    def disconnect(self) -> None:
        if not self._connected:
            return
        for name in ["L", "goni", "z", "y", "x"]:
            try:
                self._axes[name].disconnect()
            except Exception:
                pass
        self._connected = False
        print("Disconnected.")

    def home_all(self) -> None:
        self._require_ready()
        self.home(["x", "y", "z", "goni", "L"])

    def home(self, names: AxisNames) -> None:
        self._require_ready()
        names_list = [names] if isinstance(names, str) else list(names)

        for name in names_list:
            print(f"Homing {name} ...")
            if name == "L":
                self.L.move_to(0.0)
            else:
                self._axes[name].home()

        print("Home done.")

    def positions(self) -> dict[str, float]:
        self._require_ready()
        return {
            "x": self.x.position(),
            "y": self.y.position(),
            "z": self.z.position(),
            "goni": self.goni.angle(),
            "L": self.L.position_mm(),
        }

    def move_axis_to(self, axis: str, value: float) -> None:
        self._require_ready()
        axis_obj = self._get_axis(axis)
        axis_key = axis.lower()
        if axis_key == "goni":
            axis_obj.move_to(float(value))
        else:
            axis_obj.move_to(float(value))

    def move_axis_by(self, axis: str, delta: float) -> None:
        self._require_ready()
        axis_obj = self._get_axis(axis)
        axis_key = axis.lower()
        if axis_key == "goni":
            axis_obj.move_by(float(delta))
        else:
            axis_obj.move_by(float(delta))

    def jog_axis(self, axis: str, step: float) -> None:
        self._require_ready()
        axis_key = axis.lower()
        if axis_key == "l":
            self.L.move_by(float(step))
            return
        axis_obj = self._get_axis(axis)
        axis_obj.jog(float(step))

    def set_kst_speed(self, axis: str, max_vel: float, accel: float, min_vel: float = 0.0) -> None:
        self._require_ready()
        axis_obj = self._get_axis(axis)
        axis_key = axis.lower()
        if axis_key not in {"x", "y", "z"}:
            raise ValueError("KST speed config only supported on x/y/z axes.")
        axis_obj.set_speed(float(max_vel), float(accel), float(min_vel))

    def get_kst_speed(self, axis: str) -> dict:
        self._require_ready()
        axis_obj = self._get_axis(axis)
        axis_key = axis.lower()
        if axis_key not in {"x", "y", "z"}:
            raise ValueError("KST speed readout only supported on x/y/z axes.")
        return axis_obj.get_speed()

    def set_pdv_speed(self, speed: int) -> None:
        self._require_ready()
        self.L.set_speed(int(speed))

    def set_pdv_max_speed(self, max_speed: int) -> None:
        self._require_ready()
        self.L.set_max_speed(int(max_speed))

    def stop_pdv(self) -> None:
        self._require_ready()
        self.L.stop()

    def _get_axis(self, axis: str):
        axis_key = axis.lower()
        if axis_key in {"x", "y", "z", "goni", "l"}:
            return getattr(self, axis_key if axis_key != "l" else "L")
        raise ValueError(f"Unknown transfer axis '{axis}'. Use x, y, z, goni, or L.")

    def _init_hardware(self) -> None:
        if self._axes:
            return
        self.x = KST201(self._config["x_serial"], polling_ms=self._config["polling_ms"])
        self.y = KST201(self._config["y_serial"], polling_ms=self._config["polling_ms"])
        self.z = KST201(self._config["z_serial"], polling_ms=self._config["polling_ms"])
        self.goni = KDC101(self._config["goni_serial"], polling_ms=self._config["polling_ms"])
        self.L = PDVStage(
            com_port=self._config["L_com_port"],
            channel="L",
            pos_min_mm=self._config["L_pos_min_mm"],
            pos_max_mm=self._config["L_pos_max_mm"],
            max_speed=self._config["L_max_speed"],
        )
        self._axes = {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "goni": self.goni,
            "L": self.L,
        }

    def _require_ready(self) -> None:
        if not self._connected:
            raise RuntimeError("Transfer controller not connected. Call connect() first.")


class TransferController(TransferControl):
    """Compatibility alias for workflow code expecting TransferController."""

    pass
