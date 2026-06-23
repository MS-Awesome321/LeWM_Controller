# -*- coding: utf-8 -*-
"""
kz100_toolbox
-----------------
Python helper library for KZ-100 (and compatible) multi-axis motor controllers
that communicate over a serial port.

Main functions:
- SetSpeed               : continuous speed motion
- SetMaxSpeed            : set maximum speed limit
- SetDistance            : move a specified distance (in microsteps/revolutions)
- GetDisplay             : read the displayed position
- GetStatus              : read limit / motion mode / home status
- SaveRecall             : save / recall arbitrary motion data
- SetArbitraryMotionData : configure arbitrary speed motion profile
- TrigArbitraryMotion    : trigger arbitrary speed motion
- Stop                   : emergency stop for all axes

Requirements:
- pyserial:   pip install pyserial
- All functions use fixed serial parameters: 57600, 8N1, timeout=3s
"""

import serial
import time
from typing import Sequence


# ======================== Internal helper functions ======================== #

def _open_serial(COM_port: str, timeout: float = 3.0) -> serial.Serial:
    """
    Open a serial port with the controller's standard settings.
    """
    ser = serial.Serial(
        COM_port,
        baudrate=57600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=timeout,
        rtscts=False,
    )
    return ser


def _write_command(ser: serial.Serial, command: str) -> None:
    """
    Send a command string to the controller, one character at a time.
    After each character, read 1 byte back as an echo/ack.

    Example: command = 'SX100;'
    """
    for ch in command:
        ser.write(ch.encode("ascii"))  # Python 3 requires bytes
        ser.read(1)                    # read 1 byte echo (usually 0x0D)


# ======================== Public API functions ======================== #

###################  Function: set continuous speed  ###################


def SetSpeed(COM_port: str, channel: str, speed: int) -> None:
    """
    Run the motor on the specified channel at a continuous speed.

    Parameters:
        COM_port : Serial port name, e.g. 'COM1', 'COM2', 'COM3', ...
        channel  : Axis identifier, one of 'X', 'Y', 'Z', 'L'
        speed    : Speed value, integer in range -16383 ~ 16383
                Unit: (5^5 / 2^20) revolutions per second
    """
    command = f"S{channel}{speed};"

    ser = _open_serial(COM_port)
    try:
        _write_command(ser, command)
    finally:
        ser.close()


###################  Function: set maximum speed limit  ###################


def SetMaxSpeed(COM_port: str, channel: str, MaxSpeed: int) -> None:
    """
    Set the maximum running speed for the specified axis.

    Parameters:
        COM_port : Serial port name
        channel  : Axis identifier, one of 'X', 'Y', 'Z', 'L'
        MaxSpeed : Maximum speed value, 0 ~ 16383
                Unit: (5^5 / 2^20) revolutions per second
    """
    command = f"M{channel}{MaxSpeed};"

    ser = _open_serial(COM_port)
    try:
        _write_command(ser, command)
    finally:
        ser.close()


###################  Function: move a specified distance  ###################


def SetDistance(COM_port: str, channel: str, distance: int) -> None:
    """
    Move the motor on the specified axis by a given distance.

    Parameters:
        COM_port : Serial port name
        channel  : Axis identifier, one of 'X', 'Y', 'Z', 'L'
        distance : Distance value, integer in range -2^30 ~ 2^30
                Unit: 1/12800 revolution  (i.e., 1 revolution = 12800 units)
    """
    command = f"D{channel}{distance};"

    ser = _open_serial(COM_port)
    try:
        _write_command(ser, command)
    finally:
        ser.close()


###################  Function: read displayed position  ###################


def GetDisplay(COM_port: str, channel: str) -> int:
    """
    Read the displayed position data for a given axis.

    Parameters:
        COM_port : Serial port name
        channel  : Axis identifier, one of 'X', 'Y', 'Z', 'L'

    Returns:
        Integer position value decoded from 9 characters returned
        by the controller.
    """
    command = f"U{channel};"

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        ser = _open_serial(COM_port)
        try:
            _write_command(ser, command)
            s = ser.read(9)  # 9 bytes
        except Exception:
            # If serial open/write fails, we might want to retry or just let it fail.
            # For now, let's proceed to close and check 's'
            s = b""
        finally:
            ser.close()

        text = s.decode("ascii", errors="ignore").strip()
        if text:
            try:
                return int(text)
            except ValueError:
                pass
        
        # If we are here, it failed. Wait a bit and retry.
        time.sleep(0.1)

    # If all retries fail, raise a clear error
    raise IOError(f"Failed to get display data from {COM_port} (ch={channel}) after {MAX_RETRIES} attempts. Last received: '{text}'")


###################  Function: read limit/mode/home status  ###################


def GetStatus(COM_port: str, Content: str) -> str:
    """
    Read one status byte: limit status, motion mode, or home status.

    Parameters:
        COM_port : Serial port name
        Content  : 'S' -> limit status
                'M' -> motion mode
                'H' -> home status

    Returns:
        A 1-character string. Bit definitions are given in the controller
        documentation.
    """
    command = f"U{Content};"

    ser = _open_serial(COM_port)
    try:
        _write_command(ser, command)
        s = ser.read(1)  # 1 byte
    finally:
        ser.close()

    return s.decode("ascii", errors="ignore")


###################  Function: save/recall arbitrary motion data  ###################


def SaveRecall(COM_port: str, operation: str, DataID: int) -> None:
    """
    Save or recall arbitrary speed motion data between controller RAM and disk.

    Parameters:
        COM_port  : Serial port name
        operation : 'S' -> Save (RAM -> disk)
                    'R' -> Recall (disk -> RAM)
        DataID    : Arbitrary motion data ID
    """
    command = f"A{operation}{DataID};"

    ser = _open_serial(COM_port)
    try:
        _write_command(ser, command)
    finally:
        ser.close()


###################  Function: configure arbitrary speed motion data  ###################


def SetArbitraryMotionData(
    COM_port: str,
    channel: str,
    total_seg: int,
    duration: Sequence[int],
    acceleration: Sequence[int],
    repeat: int,
) -> None:
    """
    Configure arbitrary speed motion segments and repeat count.

    Parameters:
        COM_port    : Serial port name
        channel     : Axis identifier, one of 'X', 'Y', 'Z', 'L'
        total_seg   : Total number of segments
        duration    : Sequence (list/tuple) of length total_seg,
                    each entry is the duration of that segment
                    Unit: 0.4096 ms
        acceleration: Sequence (list/tuple) of length total_seg,
                    each entry is the acceleration for that segment
                    Unit: (5^12 / 2^33) revolutions per second^2
        repeat      : Repeat count, 0 means run exactly once
    """
    if len(duration) != total_seg or len(acceleration) != total_seg:
        raise ValueError("Length of duration and acceleration must equal total_seg")

    ser = _open_serial(COM_port)
    try:
        # (1) Send axis and segment count
        cmd = f"N{channel}{total_seg};"
        _write_command(ser, cmd)

        # (2) Send duration and acceleration for each segment
        for T, A in zip(duration, acceleration):
            cmd_T = f"LT{T};"
            _write_command(ser, cmd_T)

            cmd_A = f"LA{A};"
            _write_command(ser, cmd_A)

        # (3) Send repeat count
        cmd_R = f"R{channel}{repeat};"
        _write_command(ser, cmd_R)
    finally:
        ser.close()


###################  Function: trigger arbitrary speed motion  ###################


def TrigArbitraryMotion(COM_port: str, SumChannels: int) -> None:
    """
    Trigger arbitrary speed motion for selected axes.

    Parameters:
        COM_port    : Serial port name
        SumChannels : Sum of axis weights that participate in arbitrary motion.
                    X:1, Y:2, Z:4, L:8, not participating: 0.
                    Examples:
                        X only        -> 1
                        X + Y         -> 3
                        X + Y + Z + L -> 15
    """
    command = f"TM{SumChannels};"

    ser = _open_serial(COM_port)
    try:
        _write_command(ser, command)
    finally:
        ser.close()


###################  Function: emergency stop all axes  ###################


def Stop(COM_port: str) -> None:
    """
    Emergency stop for all axes.

    Parameters:
        COM_port : Serial port name
    """
    command = "PA;"

    ser = _open_serial(COM_port)
    try:
        _write_command(ser, command)
    finally:
        ser.close()


# ======================== Official-style demo (only when run directly) ======================== #

# if __name__ == "__main__":
# # #     # WARNING:
# # #     # Running this file directly will move the stages according to the
# # #     # official example sequence. Make sure:
# # #     # - The controller is connected
# # #     # - The COM port is correct
# # #     # - Motion is safe for your setup
#     COM = "COM9"  # <-- change this to your actual serial port

# # #     # Basic function usage examples (equivalent to the vendor's original script):

#     SetDistance(COM, 'Z', -30000)          # Z axis: 1 revolution (1 rev = 12800 microsteps)
#     SetDistance(COM, 'Y', -10000)         # Y axis: 10 revolutions

#     # SetSpeed(COM, 'Y', -1000)              # X axis: ~6 rev/s continuous
#     # time.sleep(3)
# # #     SetSpeed(COM, 'L', 4027)              # L axis: ~12 rev/s continuous

# # #     SetMaxSpeed(COM, 'Z', 2013)           # Z axis: max speed ~6 rev/s
# # #     SetMaxSpeed(COM, 'L', 1007)           # L axis: max speed ~3 rev/s

# # #     print(GetDisplay(COM, 'Y'))           # Read and print Y axis position
# # #     print(GetDisplay(COM, 'L'))           # Read and print L axis position

# # #     print(GetStatus(COM, 'S'))            # Limit status
# # #     print(GetStatus(COM, 'M'))            # Motion mode status
# # #     print(GetStatus(COM, 'H'))            # Home status
# #     # Co
if __name__ == '__main__':
    print(GetStatus('com7','M'))