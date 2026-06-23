"""Camera controller for the Qumus processing hardware.

This module provides a complete, self-contained camera controller implementation
supporting both Uvcham SDK and OpenCV fallback modes.
"""

from __future__ import annotations

import os, sys
import ctypes

UVCHAM_DIR = r"C:\Users\Qumus_PC\Documents\Qumus"
IS_WINDOWS = os.name == "nt"

if IS_WINDOWS and UVCHAM_DIR not in sys.path:
    sys.path.insert(0, UVCHAM_DIR)

if IS_WINDOWS and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(UVCHAM_DIR)
elif IS_WINDOWS:
    os.environ["PATH"] = UVCHAM_DIR + os.pathsep + os.environ.get("PATH", "")

import threading
import time
from typing import Optional

import cv2
import numpy as np
try:
    import uvcham as uvcham
except Exception:
    uvcham = None

try:
    import pythoncom
except Exception:
    pythoncom = None

class CameraController:
    """Camera controller with dual-mode support (Uvcham SDK + OpenCV fallback).
    
    This class provides:
    - Camera initialization and frame capture
    - Live view window
    - Focus detection and auto-focus capabilities
    - Photo capture with quality metrics
    """

    def __init__(self, index: int = 0, fps: int = 15, window_name: str = "Live View") -> None:
        """Initialize the camera controller.
        
        Args:
            index: Camera device index (default: 0)
            fps: Target frames per second (default: 15)
            window_name: Name for the live view window (default: "Live View")
        """
        self.cam_index = index
        self.fps = max(1, int(fps))
        self.window_name = window_name
        self.counter = 0

        # Camera handles
        self._cam = None  # Uvcham handle
        self._cap = None  # OpenCV handle (fallback)
        self._is_opencv = False

        # Frame buffer
        self._buf = None
        self._width = None
        self._height = None
        self._stride = None

        # Frame synchronization
        self._latest = None
        self._lock = threading.Lock()

        # Thread management
        self._running = False
        self._grab_thread = None

        # Live view state
        self._preview_thread = None
        self._preview_running = False
        self._window_inited = False
        self._init_window_size = (960, 540)  # Default initial size (W, H)
        self._shared_frame_path = os.getenv("AGENT_SHARED_FRAME_PATH") or None
        self._shared_frame_interval_s = float(os.getenv("AGENT_SHARED_FRAME_INTERVAL_S", "0.2"))
        self._last_shared_frame_t = 0.0

    # ========== Lifecycle methods ========== #

    def start(self) -> None:
        """Initialize and start the camera."""
        if self._running:
            return

        # Try Uvcham first
        use_uvcham = False
        cams = []

        if IS_WINDOWS and uvcham is not None:
            if pythoncom:
                try:
                    pythoncom.CoInitialize()
                except Exception:
                    pass
            if hasattr(ctypes, "windll"):
                try:
                    COINIT_APARTMENTTHREADED = 0x2
                    ctypes.windll.ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
                except Exception:
                    pass
            try:
                cams = uvcham.Uvcham.enum()
            except Exception:
                cams = []

        if cams and self.cam_index < len(cams):
            use_uvcham = True
        else:
            print(f"[CameraController] No Uvcham devices found or index {self.cam_index} out of range.") 
            print(f"[CameraController] Attempting OpenCV fallback for index {self.cam_index}...")

        if use_uvcham:
            # === Uvcham initialization ===
            self._is_opencv = False
            dev = cams[self.cam_index]
            self._cam = uvcham.Uvcham.open(dev.id)
            if self._cam is None:
                raise RuntimeError("Failed to open camera (Uvcham)")

            # Set BGR format for OpenCV compatibility
            self._cam.put(uvcham.UVCHAM_FORMAT, 0)
            res = self._cam.get(uvcham.UVCHAM_RES)
            self._width = self._cam.get(uvcham.UVCHAM_WIDTH | res)
            self._height = self._cam.get(uvcham.UVCHAM_HEIGHT | res)

            self._stride = uvcham.TDIBWIDTHBYTES(self._width * 24)
            self._buf = bytes(self._stride * self._height)

            self._running = True
            self._cam.start(None, self._callback, self)
            print(f"[CameraController] Started Uvcham: {dev.displayname}")

        else:
            # === OpenCV fallback initialization ===
            self._is_opencv = True
            if IS_WINDOWS:
                self._cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)  # DirectShow on Windows
            else:
                self._cap = cv2.VideoCapture(self.cam_index)
            if not self._cap.isOpened() and IS_WINDOWS:
                self._cap = cv2.VideoCapture(self.cam_index)
            
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open camera via OpenCV at index {self.cam_index}")

            # Try to set high resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            # Read actual properties
            self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._running = True
            print(f"[CameraController] Started OpenCV Camera (Index {self.cam_index}, {self._width}x{self._height})")

        # Start frame grabbing thread
        self._grab_thread = threading.Thread(target=self._grab_loop, daemon=False)
        self._grab_thread.start()

    def stop(self) -> None:
        """Stop and release the camera."""
        self.close_live_view()

        self._running = False
        if self._grab_thread and self._grab_thread.is_alive():
            self._grab_thread.join(timeout=1.0)

        # Stop Uvcham
        if self._cam:
            try:
                self._cam.close()
            except Exception:
                pass
            self._cam = None
        
        # Stop OpenCV
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        self._buf = None
        self._latest = None
        self._window_inited = False

    # ========== Internal methods ========== #

    @staticmethod
    def _callback(event, ctx):
        """Uvcham callback (frames are pulled in grab loop)."""
        return

    def _grab_loop(self) -> None:
        """Background thread for continuous frame capture."""
        period = 1.0 / self.fps
        while self._running:
            t0 = time.time()
            img = None

            if not self._is_opencv:
                # === Uvcham frame pull ===
                try:
                    if self._cam:
                        self._cam.pull(self._buf)
                        raw = np.frombuffer(self._buf, np.uint8).reshape(self._height, self._stride)
                        raw = raw[:, : self._width * 3]
                        img = raw.reshape(self._height, self._width, 3)  # BGR
                except Exception:
                    pass
            else:
                # === OpenCV frame read ===
                try:
                    if self._cap:
                        ret, frame = self._cap.read()
                        if ret:
                            img = frame
                except Exception:
                    pass

            # Update latest frame
            if img is not None:
                with self._lock:
                    self._latest = img.copy()
                if self._shared_frame_path:
                    now = time.time()
                    if now - self._last_shared_frame_t >= self._shared_frame_interval_s:
                        self._write_shared_frame(img)
                        self._last_shared_frame_t = now

            # Sleep for remainder of period
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    def _set_autofocus_mode(self, mode: int) -> None:
        """
        Internal: set autofocus mode via uvcham.UVCHAM_AFMODE.
        mode:
          0 = manual focus (AF off)
          1 = continuous autofocus
          2 = one-shot autofocus
        """
        if not self._running or self._cam is None:
            raise RuntimeError("Camera not started. Call start() first.")
        if self._is_opencv:
            raise RuntimeError("Autofocus not available in OpenCV fallback mode.")

        mode = int(mode)

        try:
            self._cam.put(uvcham.UVCHAM_AFMODE, mode)
            return
        except Exception:
            pass

        was_preview = bool(self._preview_thread and self._preview_thread.is_alive())
        try:
            if was_preview:
                self.close_live_view()

            self._running = False
            try:
                if self._grab_thread and self._grab_thread.is_alive():
                    self._grab_thread.join(timeout=1.0)
            except Exception:
                pass

            try:
                if self._cam:
                    self._cam.stop()
            except Exception:
                pass

            self._cam.put(uvcham.UVCHAM_AFMODE, mode)

            self._running = True
            self._cam.start(None, self._callback, self)
            self._grab_thread = threading.Thread(target=self._grab_loop, daemon=False)
            self._grab_thread.start()

            if was_preview:
                self.open_live_view()
        except Exception as exc:
            raise RuntimeError(f"Failed to set autofocus mode={mode}: {exc}") from exc

    def _get_latest_frame(self) -> np.ndarray | None:
        """Get the most recent frame."""
        with self._lock:
            return None if self._latest is None else cv2.flip(self._latest.copy(), 0)

    # ========== Live view methods ========== #

    def set_live_view_size(self, width: int, height: int) -> None:
        """Set the live view window size.
        
        Args:
            width: Window width in pixels
            height: Window height in pixels
        """
        self._init_window_size = (int(width), int(height))
        if self._window_inited:
            try:
                cv2.resizeWindow(self.window_name, self._init_window_size[0], self._init_window_size[1])
            except Exception:
                pass

    def open_live_view(self) -> None:
        """Open the live view window (non-blocking)."""
        if not self._running:
            raise RuntimeError("Camera not started. Call start() first.")

        if self._preview_thread and self._preview_thread.is_alive():
            return  # Already running

        self._preview_running = True

        def _preview_loop():
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._window_inited = True
            cv2.resizeWindow(self.window_name, self._init_window_size[0], self._init_window_size[1])

            while self._preview_running and self._running:
                frame = self._get_latest_frame()
                if frame is not None:
                    cv2.imshow(self.window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):  # ESC or q
                    break

            # Close window
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            self._window_inited = False
            self._preview_running = False

        self._preview_thread = threading.Thread(target=_preview_loop, daemon=False)
        self._preview_thread.start()

    def close_live_view(self) -> None:
        """Close the live view window."""
        self._preview_running = False
        if self._preview_thread and self._preview_thread.is_alive():
            self._preview_thread.join(timeout=1.0)
        self._preview_thread = None

    def _write_shared_frame(self, frame: np.ndarray) -> None:
        if not self._shared_frame_path:
            return
        try:
            target = self._shared_frame_path
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            tmp_path = f"{target}.tmp"
            if cv2.imwrite(tmp_path, frame):
                os.replace(tmp_path, target)
        except Exception:
            pass

    # ========== Photo capture methods ========== #

    def take_photo(self, path: str, timeout: float = 2.0) -> str:
        """Capture and save a photo.
        
        Args:
            path: File path to save the image
            timeout: Maximum wait time for frame availability
            
        Returns:
            Absolute path to the saved image
        """
        if not self._running:
            raise RuntimeError("Camera not started. Call start() first.")

        t0 = time.time()
        while True:
            frame = self._get_latest_frame()
            if frame is not None:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                ok = cv2.imwrite(path,frame)
                if not ok:
                    raise IOError(f"Failed to write image: {path}")
                return os.path.abspath(path)

            if time.time() - t0 > timeout:
                raise TimeoutError("No frame available (take_photo timeout)")

            time.sleep(0.02)

    def snap(self, timeout: float = 2.0) -> np.ndarray:
        """Capture a frame without saving to disk.
        
        Args:
            timeout: Maximum wait time for frame availability
            
        Returns:
            Frame as numpy array (BGR format)
        """
        if not self._running:
            raise RuntimeError("Camera not started. Call start() first.")

        t0 = time.time()
        while True:
            frame = self._get_latest_frame()
            if frame is not None:
                return frame  # BGR uint8

            if time.time() - t0 > timeout:
                raise TimeoutError("No frame available (snap timeout)")

            time.sleep(0.02)

    # ========== Focus quality methods ========== #

    def blur_score_bgr(self, bgr: np.ndarray, shrink: int = 2) -> float:
        """Calculate focus quality metric (higher = sharper).

        Uses Tenengrad (Sobel gradient energy) for better stability on microscopy images.

        Args:
            bgr: Input image in BGR format
            shrink: Downsample factor for speed (default: 2)

        Returns:
            Focus score (log of mean gradient energy)
        """
        if bgr is None:
            return float("-inf")

        bgr = np.asarray(bgr)
        if bgr.ndim == 3:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = bgr

        if shrink and shrink > 1:
            h, w = gray.shape[:2]
            gray = cv2.resize(
                gray,
                (max(1, w // shrink), max(1, h // shrink)),
                interpolation=cv2.INTER_AREA,
            )

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        e = float(np.mean(gx * gx + gy * gy))
        return float(np.log(e + 1e-12))

    def blur_score_bgr_laplacian(self, bgr: np.ndarray, shrink: int = 2) -> float:
        """Legacy focus metric for comparison (log variance of Laplacian)."""
        if bgr is None:
            return float("-inf")

        bgr = np.asarray(bgr)
        if bgr.ndim == 3:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = bgr

        if shrink and shrink > 1:
            h, w = gray.shape[:2]
            gray = cv2.resize(
                gray,
                (max(1, w // shrink), max(1, h // shrink)),
                interpolation=cv2.INTER_AREA,
            )

        lap = cv2.Laplacian(gray, cv2.CV_32F)
        v = float(lap.var())
        return float(np.log(v + 1e-12))

    def wait_focus_and_snap(
        self,
        min_blur: float = 3.5,
        max_checks: int = 100,
        interval_s: float = 0.02,
        timeout_per_snap: float = 2.0,
        flush: int = 2,
        shrink: int = 2,
    ) -> tuple[np.ndarray | None, float]:
        """Wait for focus and capture a frame.
        
        Args:
            min_blur: Minimum acceptable focus score
            max_checks: Maximum number of focus checks
            interval_s: Time between focus checks
            timeout_per_snap: Timeout for each frame capture
            flush: Number of frames to flush before checking focus
            shrink: Downsample factor for focus calculation
            
        Returns:
            Tuple of (frame, focus_score)
        """
        if not self._running:
            raise RuntimeError("Camera not started. Call start() first.")

        best_score = float("-inf")
        best_bgr = None

        for _ in range(int(max_checks)):
            bgr = None

            # Flush old frames to reduce latency
            for _k in range(max(1, int(flush))):
                bgr = self.snap(timeout=5.0)
                time.sleep(0.005)

            if bgr is None:
                time.sleep(interval_s)
                continue

            score = self.blur_score_bgr(bgr, shrink=shrink)

            if score > best_score:
                best_score = score
                best_bgr = bgr

            if score >= float(min_blur):
                return bgr, score

            time.sleep(interval_s)

        return best_bgr, best_score

    def wait_focus_and_take_photo(
        self,
        path: str,
        min_blur: float = 3.5,
        max_checks: int = 100,
        interval_s: float = 0.02,
        timeout_per_snap: float = 2.0,
        flush: int = 2,
        shrink: int = 2,
        save_path: Optional[str] = None,
    ) -> tuple[str | None, float]:
        """Wait for focus and save a photo.
        
        Args:
            path: Directory to save the image
            min_blur: Minimum acceptable focus score
            max_checks: Maximum number of focus checks
            interval_s: Time between focus checks
            timeout_per_snap: Timeout for each frame capture
            flush: Number of frames to flush before checking focus
            shrink: Downsample factor for focus calculation
            
        Returns:
            Tuple of (absolute_path, focus_score)
        """
        bgr, score = self.wait_focus_and_snap(
            min_blur=min_blur,
            max_checks=max_checks,
            interval_s=interval_s,
            timeout_per_snap=timeout_per_snap,
            flush=flush,
            shrink=shrink,
        )

        if bgr is None:
            return None, score

        if save_path is not None:
            out_path = save_path
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)
            out_path = os.path.join(path, f"test_{self.counter}.jpg")
            self.counter += 1

        ok = cv2.imwrite(out_path, bgr)
        if not ok:
            raise IOError(f"Failed to write image: {out_path}")
        abs_path = os.path.abspath(out_path)

        if score >= float(min_blur):
            print(f"[focus] HIT saved: {abs_path}  score={score:.3f} (>= {min_blur})")
        else:
            print(f"[focus] NOT HIT, saved BEST anyway: {abs_path}  best_score={score:.3f} (< {min_blur})")

        return abs_path, score

    def manual_focus_and_take_photo(
        self,
        focus,
        path: str,
        min_blur: float = 4.0,
        start_offset_um: int = -500,
        end_offset_um: int = 500,
        step_um: int = 20,
        timeout: float = 2.0,
        settle_s: float = 0.05,
        shrink: int = 2,
        wait_factor_s_per_um: Optional[float] = None,
        save_path: Optional[str] = None,
    ) -> tuple[str, float]:
        """
        Manual focus + take photo (NO autofocus).

        Scans focus offsets from start_offset_um to end_offset_um (exclusive) by step_um.
        Uses only focus.move_relative_um(). Always returns focus to the entry position.
        """
        if not getattr(self, "_running", False):
            raise RuntimeError("Camera is not running. Call camera.start() first.")

        if wait_factor_s_per_um is None:
            wait_factor_s_per_um = getattr(focus, "wait_factor_s_per_um", 0.0)

        offsets = list(range(int(start_offset_um), int(end_offset_um), int(step_um)))
        if (start_offset_um, end_offset_um, step_um) == (-500, 500, 20) and len(offsets) != 50:
            raise RuntimeError(f"Unexpected offsets length: {len(offsets)} (expected 50).")

        best_score = float("-inf")
        best_bgr = None
        hit = False
        current_offset = 0.0

        def _wait(dist_um: float) -> None:
            if settle_s and settle_s > 0:
                time.sleep(float(settle_s))
            if wait_factor_s_per_um and wait_factor_s_per_um > 0:
                time.sleep(float(wait_factor_s_per_um) * abs(float(dist_um)))

        try:
            bgr0 = self.snap(timeout=timeout)
            score0 = float(self.blur_score_bgr(bgr0, shrink=shrink))
            if score0 >= float(min_blur):
                if save_path is not None:
                    out_path = save_path
                    out_dir = os.path.dirname(out_path)
                    if out_dir:
                        os.makedirs(out_dir, exist_ok=True)
                else:
                    os.makedirs(path, exist_ok=True)
                    out_path = os.path.join(path, f"test_{self.counter}.jpg")
                    self.counter += 1

                if not cv2.imwrite(out_path, bgr0):
                    raise IOError(f"Failed to write image: {out_path}")
                abs_path = os.path.abspath(out_path)
                print(f"[manual_focus] ORIGIN HIT: score={score0:.3f} (>= {min_blur}) -> {abs_path}")
                return abs_path, float(score0)

            for offset in offsets:
                delta = float(offset) - float(current_offset)
                if abs(delta) > 0:
                    focus.move_relative_um(delta)
                    _wait(delta)
                    current_offset = float(offset)

                bgr = self.snap(timeout=timeout)
                score = float(self.blur_score_bgr(bgr, shrink=shrink))

                if score > best_score:
                    best_score = score
                    best_bgr = bgr

                if score >= float(min_blur):
                    hit = True
                    best_score = score
                    best_bgr = bgr
                    break

            if best_bgr is None:
                raise RuntimeError("No frames captured during manual focus scan.")

            if save_path is not None:
                out_path = save_path
                out_dir = os.path.dirname(out_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
            else:
                os.makedirs(path, exist_ok=True)
                out_path = os.path.join(path, f"test_{self.counter}.jpg")
                self.counter += 1

            if not cv2.imwrite(out_path, best_bgr):
                raise IOError(f"Failed to write image: {out_path}")
            abs_path = os.path.abspath(out_path)

            if hit:
                print(f"[manual_focus] HIT: score={best_score:.3f} (>= {min_blur}) -> {abs_path}")
            else:
                print(f"[manual_focus] BEST: score={best_score:.3f} (< {min_blur}) -> {abs_path}")

            return abs_path, float(best_score)
        finally:
            try:
                back = -float(current_offset)
                if abs(back) > 0:
                    focus.move_relative_um(back)
                    _wait(back)
            except Exception as exc:
                print(f"[manual_focus] WARN: failed to return to origin: {exc}")

    def scan_focus_and_take_photo(
        self,
        focus,
        path: str,
        min_blur: float = 7.0,
        start_offset_um: int = -300,
        end_offset_um: int = 300,
        step_um: int = 20,
        timeout: float = 2.0,
        settle_s: float = 0.05,
        shrink: int = 2,
        wait_factor_s_per_um: Optional[float] = None,
        save_path: Optional[str] = None,
    ) -> tuple[str, float]:
        """
        Manual focus + take photo (NO autofocus). Stays at best focus position.

        Returns:
            (abs_path, score)
        """
        if not getattr(self, "_running", False):
            raise RuntimeError("Camera is not running. Call camera.start() first.")

        if wait_factor_s_per_um is None:
            wait_factor_s_per_um = getattr(focus, "wait_factor_s_per_um", 0.0)

        offsets = list(range(int(start_offset_um), int(end_offset_um), int(step_um)))
        if (start_offset_um, end_offset_um, step_um) == (-300, 300, 20) and len(offsets) != 30:
            raise RuntimeError(f"Unexpected offsets length: {len(offsets)} (expected 30).")

        best_score = float("-inf")
        best_bgr = None
        best_offset = 0.0
        hit = False

        current_offset = 0.0

        def _wait(dist_um: float) -> None:
            if settle_s and settle_s > 0:
                time.sleep(float(settle_s))
            if wait_factor_s_per_um and wait_factor_s_per_um > 0:
                time.sleep(float(wait_factor_s_per_um) * abs(float(dist_um)))

        bgr0 = self.snap(timeout=timeout)
        score0 = float(self.blur_score_bgr(bgr0, shrink=shrink))

        best_score = score0
        best_bgr = bgr0
        best_offset = 0.0

        if score0 >= float(min_blur):
            if save_path is not None:
                out_path = save_path
                out_dir = os.path.dirname(out_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
            else:
                os.makedirs(path, exist_ok=True)
                out_path = os.path.join(path, f"test_{self.counter}.jpg")
                self.counter += 1

            if not cv2.imwrite(out_path, bgr0):
                raise IOError(f"Failed to write image: {out_path}")
            abs_path = os.path.abspath(out_path)

            print(f"[manual_focus] ORIGIN HIT: score={score0:.3f} (>= {min_blur}) -> {abs_path}")
            return abs_path, float(score0)

        for offset in offsets:
            delta = float(offset) - float(current_offset)
            if abs(delta) > 0:
                focus.move_relative_um(delta)
                _wait(delta)
                current_offset = float(offset)

            bgr = self.snap(timeout=timeout)
            score = float(self.blur_score_bgr(bgr, shrink=shrink))

            if score > best_score:
                best_score = score
                best_bgr = bgr
                best_offset = float(current_offset)

            if score >= float(min_blur):
                hit = True
                best_score = score
                best_bgr = bgr
                best_offset = float(current_offset)
                break

        if best_bgr is None:
            raise RuntimeError("No frames captured during manual focus scan.")

        if save_path is not None:
            out_path = save_path
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)
            out_path = os.path.join(path, f"test_{self.counter}.jpg")
            self.counter += 1

        if not cv2.imwrite(out_path, best_bgr):
            raise IOError(f"Failed to write image: {out_path}")
        abs_path = os.path.abspath(out_path)

        if hit:
            print(f"[manual_focus] HIT: best_score={best_score:.3f} (>= {min_blur}) -> {abs_path}")
        else:
            print(f"[manual_focus] BEST: best_score={best_score:.3f} (< {min_blur}) -> {abs_path}")

        delta_to_best = float(best_offset) - float(current_offset)
        if abs(delta_to_best) > 0:
            focus.move_relative_um(delta_to_best)
            _wait(delta_to_best)

        return abs_path, float(best_score)

    def autofocus_on(self) -> None:
        """Enable continuous autofocus."""
        self._set_autofocus_mode(1)

    def autofocus_off(self) -> None:
        """Disable autofocus (manual focus mode)."""
        self._set_autofocus_mode(0)

    def autofocus_once(self) -> None:
        """Trigger one-shot autofocus."""
        self._set_autofocus_mode(2)

    def get_af_zone_info(self) -> dict:
        """
        Query AF zone grid size and current zone.
        Returns dict: {"min": mn, "max": mx, "default": df, "w": w, "h": h, "current": cur}
        """
        if not self._running or self._cam is None:
            raise RuntimeError("Camera not started. Call start() first.")
        if self._is_opencv:
            raise RuntimeError("Autofocus not available in OpenCV fallback mode.")
        mn, mx, df = self._cam.range(uvcham.UVCHAM_AFZONE)
        w = mx & 0xFF
        h = (mx >> 8) & 0xFF
        cur = self._cam.get(uvcham.UVCHAM_AFZONE)
        return {"min": mn, "max": mx, "default": df, "w": w, "h": h, "current": cur}

    def set_af_zone(self, zone: int | None = None, row: int | None = None, col: int | None = None, where: str | None = None) -> dict:
        """
        Set autofocus zone on a discrete w*h grid (NOT arbitrary pixel ROI).
        Choose one of: zone, row/col, or where in {"center","tl","tr","bl","br"}.
        """
        if not self._running or self._cam is None:
            raise RuntimeError("Camera not started. Call start() first.")
        if self._is_opencv:
            raise RuntimeError("Autofocus not available in OpenCV fallback mode.")

        mn, mx, df = self._cam.range(uvcham.UVCHAM_AFZONE)
        w = mx & 0xFF
        h = (mx >> 8) & 0xFF
        if w <= 0 or h <= 0:
            raise RuntimeError(f"AFZONE grid not available (w={w}, h={h}).")

        if where is not None:
            key = str(where).lower().strip()
            if key in ("center", "c", "mid", "middle"):
                row, col = h // 2, w // 2
            elif key in ("tl", "topleft", "lefttop"):
                row, col = 0, 0
            elif key in ("tr", "topright", "righttop"):
                row, col = 0, w - 1
            elif key in ("bl", "bottomleft", "leftbottom"):
                row, col = h - 1, 0
            elif key in ("br", "bottomright", "rightbottom"):
                row, col = h - 1, w - 1
            else:
                raise ValueError(f"Unknown where={where!r}.")

        if zone is None:
            if row is None or col is None:
                raise ValueError("Provide zone or (row, col) or where.")
            row = int(row)
            col = int(col)
            if not (0 <= row < h and 0 <= col < w):
                raise ValueError(f"(row,col)=({row},{col}) out of range.")
            zone = row * w + col
        else:
            zone = int(zone)

        max_zone = w * h - 1
        if not (0 <= zone <= max_zone):
            raise ValueError(f"zone={zone} out of range. Expected 0..{max_zone} (grid {w}x{h}).")

        self._cam.put(uvcham.UVCHAM_AFZONE, zone)
        cur = self._cam.get(uvcham.UVCHAM_AFZONE)
        return {"w": w, "h": h, "set": zone, "current": cur}

    def get_af_status_code(self) -> int:
        """Get autofocus feedback code (raw int)."""
        if not self._running or self._cam is None:
            raise RuntimeError("Camera not started. Call start() first.")
        if self._is_opencv:
            raise RuntimeError("Autofocus not available in OpenCV fallback mode.")
        return int(self._cam.get(uvcham.UVCHAM_AFFEEDBACK))

    def get_af_status(self) -> dict:
        """Get autofocus feedback as readable text."""
        code = self.get_af_status_code()
        mapping = {
            0: "unknown",
            1: "focused",
            2: "focusing",
            3: "out_of_focus",
            4: "up",
            5: "down",
            6: "left",
            7: "right",
        }
        return {"code": code, "status": mapping.get(code, f"code_{code}")}

    def get_focus_position(self) -> int:
        """Get manual focus position via UVCHAM_AFPOSITION."""
        if self._cam is None:
            raise RuntimeError("Camera not initialized. Call start() first.")
        if self._is_opencv:
            raise RuntimeError("Autofocus not available in OpenCV fallback mode.")
        return int(self._cam.get(uvcham.UVCHAM_AFPOSITION))

    def set_focus_position(self, pos: int, clamp: bool = True) -> dict:
        """
        Set focus position via UVCHAM_AFPOSITION.
        Forces AF off (manual mode) first.
        """
        if self._cam is None:
            raise RuntimeError("Camera not initialized. Call start() first.")
        if self._is_opencv:
            raise RuntimeError("Autofocus not available in OpenCV fallback mode.")

        pos = int(pos)
        try:
            self._set_autofocus_mode(0)
        except Exception:
            pass

        if clamp:
            try:
                mn, mx, _df = self._cam.range(uvcham.UVCHAM_AFPOSITION)
                pos = max(int(mn), min(int(mx), pos))
            except Exception:
                pos = max(0, min(854, pos))

        self._cam.put(uvcham.UVCHAM_AFPOSITION, pos)
        return {"position": pos}

    def wait(self) -> None:
        """Block main thread until Ctrl+C is pressed."""
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            return

# if __name__ == "__main__":
#     # Simple test routine
#     cam = CameraController(index=0, fps=15)
#     cam.start()
#     cam.open_live_view()
#     print("Camera started. Press Ctrl+C to exit.")
#     cam.set_focus_position(400)
#     print("Focus position:", cam.get_focus_position())
#     time.sleep(10)  # Run for 10 seconds
#     cam.stop()
#     print("Camera stopped.")
