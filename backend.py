"""OS-specific mouse injection backends.

Pynput works on Windows, macOS, and Linux X11.
Linux Wayland requires a virtual device through evdev/uinput.
"""

from __future__ import annotations

import os
import platform
from typing import Protocol


class MouseBackend(Protocol):
    supports_absolute: bool
    screen_width: int
    screen_height: int

    def move_to(self, x: float, y: float) -> None: ...
    def move_by(self, dx: float, dy: float) -> None: ...
    def press(self) -> None: ...
    def release(self) -> None: ...
    def scroll(self, dx: int, dy: int) -> None: ...
    def close(self) -> None: ...


class PynputBackend:
    supports_absolute = True

    def __init__(self) -> None:
        from pynput.mouse import Button, Controller

        self._button = Button.left
        self._mouse = Controller()
        self._remainder_x = 0.0
        self._remainder_y = 0.0
        # The tkinter screen dimensions generally use the same logical desktop
        # coordinate system as pynput. This targets the primary display only.
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            self.screen_width = root.winfo_screenwidth()
            self.screen_height = root.winfo_screenheight()
            root.destroy()
        except Exception as exc:
            raise RuntimeError(
                "Could not determine screen size using tkinter. Install the "
                "Tk support for your Python distribution or use --mapping relative."
            ) from exc

    def move_to(self, x: float, y: float) -> None:
        self._mouse.position = (round(x), round(y))

    def move_by(self, dx: float, dy: float) -> None:
        self._remainder_x += dx
        self._remainder_y += dy
        mx, my = int(self._remainder_x), int(self._remainder_y)
        self._remainder_x -= mx
        self._remainder_y -= my
        if mx or my:
            self._mouse.move(mx, my)

    def press(self) -> None:
        self._mouse.press(self._button)

    def release(self) -> None:
        self._mouse.release(self._button)

    def scroll(self, dx: int, dy: int) -> None:
        if dx or dy:
            self._mouse.scroll(dx, dy)

    def close(self) -> None:
        pass


class UInputBackend:
    """Linux-only virtual mouse for Wayland via /dev/uinput.

    Requires explicit OS permission for /dev/uinput and python-evdev.
    Cursor movement is relative, not absolute.
    """

    supports_absolute = False
    screen_width = 0
    screen_height = 0

    def __init__(self) -> None:
        if platform.system() != "Linux":
            raise RuntimeError("The uinput backend is Linux-only.")
        try:
            from evdev import UInput, ecodes
        except ImportError as exc:
            raise RuntimeError(
                "Install Linux Wayland dependency: pip install -r requirements-wayland.txt"
            ) from exc

        self._ecodes = ecodes
        try:
            self._device = UInput(
                {
                    ecodes.EV_KEY: [ecodes.BTN_LEFT],
                    ecodes.EV_REL: [
                        ecodes.REL_X, ecodes.REL_Y,
                        ecodes.REL_WHEEL, ecodes.REL_HWHEEL,
                    ],
                },
                name="Air Mouse Virtual Pointer",
            )
        except (OSError, PermissionError) as exc:
            raise RuntimeError(
                "Cannot open /dev/uinput. Load the uinput kernel module and grant "
                "your user access to /dev/uinput; see README.md. "
                "Do not run the entire camera application as root."
            ) from exc

        self._remainder_x = 0.0
        self._remainder_y = 0.0

    def move_to(self, x: float, y: float) -> None:
        raise NotImplementedError("uinput only supports relative cursor movement")

    def move_by(self, dx: float, dy: float) -> None:
        # Preserve fractional motion from small hand movements.
        self._remainder_x += dx
        self._remainder_y += dy
        mx, my = int(self._remainder_x), int(self._remainder_y)
        self._remainder_x -= mx
        self._remainder_y -= my
        if mx:
            self._device.write(self._ecodes.EV_REL, self._ecodes.REL_X, mx)
        if my:
            self._device.write(self._ecodes.EV_REL, self._ecodes.REL_Y, my)
        if mx or my:
            self._device.syn()

    def press(self) -> None:
        self._device.write(self._ecodes.EV_KEY, self._ecodes.BTN_LEFT, 1)
        self._device.syn()

    def release(self) -> None:
        self._device.write(self._ecodes.EV_KEY, self._ecodes.BTN_LEFT, 0)
        self._device.syn()

    def scroll(self, dx: int, dy: int) -> None:
        if dx:
            self._device.write(self._ecodes.EV_REL, self._ecodes.REL_HWHEEL, dx)
        if dy:
            self._device.write(self._ecodes.EV_REL, self._ecodes.REL_WHEEL, dy)
        if dx or dy:
            self._device.syn()

    def close(self) -> None:
        self._device.close()


def create_backend(name: str = "auto") -> MouseBackend:
    if name not in ("auto", "pynput", "uinput"):
        raise ValueError(f"Unknown backend {name}")
    if name == "auto":
        name = (
            "uinput" if platform.system() == "Linux"
            and os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
            else "pynput"
        )
    return UInputBackend() if name == "uinput" else PynputBackend()
