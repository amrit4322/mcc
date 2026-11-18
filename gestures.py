"""Hand gesture state machine, smoothing, scrolling, and safety behavior.

All points are MediaPipe-normalized coordinates. This file intentionally has no
imports of cv2, mediapipe, tkinter, or OS-specific mouse libraries, so it can
be unit-tested without a camera or desktop session.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional, Sequence


# MediaPipe hand landmark indexes.
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20


@dataclass
class Settings:
    pinch_down_ratio: float = 0.30
    pinch_up_ratio: float = 0.48
    pinch_confirm_s: float = 0.065
    pinch_release_s: float = 0.06
    palm_hold_s: float = 0.9
    screen_margin: float = 0.13
    cursor_gain: float = 2.4
    scroll_gain: float = 1.0
    scroll_pixels_per_tick: float = 19.0
    scroll_deadzone_px: float = 1.5
    min_cutoff: float = 1.2
    filter_beta: float = 0.007
    detection_timeout_s: float = 0.20


@dataclass
class Status:
    gesture: str
    paused: bool
    button_down: bool
    pinch_ratio: float | None = None
    pause_progress: float = 0.0


def distance(a: Any, b: Any) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def is_extended(pts: Sequence[Any], mcp: int, pip: int, tip: int) -> bool:
    """Rotation-invariant fingertip extension heuristic."""
    base = max(distance(pts[pip], pts[mcp]), 1e-6)
    return distance(pts[tip], pts[mcp]) > 1.65 * base


def finger_states(pts: Sequence[Any]) -> tuple[bool, bool, bool, bool]:
    return (
        is_extended(pts, INDEX_MCP, INDEX_PIP, INDEX_TIP),
        is_extended(pts, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
        is_extended(pts, RING_MCP, RING_PIP, RING_TIP),
        is_extended(pts, PINKY_MCP, PINKY_PIP, PINKY_TIP),
    )


def pinch_ratio(pts: Sequence[Any]) -> float:
    # Normalize using knuckle-to-knuckle palm width: camera distance invariant.
    palm_width = max(distance(pts[INDEX_MCP], pts[PINKY_MCP]), 1e-6)
    return distance(pts[THUMB_TIP], pts[INDEX_TIP]) / palm_width


def clamped(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class LowPass:
    def __init__(self):
        self.value: Optional[float] = None

    def filter(self, current: float, alpha: float) -> float:
        if self.value is None:
            self.value = current
        else:
            self.value = alpha * current + (1.0 - alpha) * self.value
        return self.value


class OneEuro2D:
    """One Euro filter (smaller jitter at rest, faster when moving)."""

    def __init__(self, min_cutoff: float, beta: float):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.x = LowPass()
        self.y = LowPass()
        self.dx = LowPass()
        self.dy = LowPass()
        self.previous_x: float | None = None
        self.previous_y: float | None = None
        self.previous_time: float | None = None

    def reset(self) -> None:
        self.__init__(self.min_cutoff, self.beta)

    @staticmethod
    def alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, y: float, now: float) -> tuple[float, float]:
        if self.previous_time is None:
            self.previous_time = now
            self.previous_x, self.previous_y = x, y
            self.x.filter(x, 1.0)
            self.y.filter(y, 1.0)
            return x, y
        dt = clamped(now - self.previous_time, 1e-3, 0.15)
        vx = (x - self.previous_x) / dt
        vy = (y - self.previous_y) / dt
        d_alpha = self.alpha(dt, 1.0)
        vx = self.dx.filter(vx, d_alpha)
        vy = self.dy.filter(vy, d_alpha)
        filtered_x = self.x.filter(x, self.alpha(dt, self.min_cutoff + self.beta * abs(vx)))
        filtered_y = self.y.filter(y, self.alpha(dt, self.min_cutoff + self.beta * abs(vy)))
        self.previous_time = now
        self.previous_x, self.previous_y = x, y
        return filtered_x, filtered_y


class GestureEngine:
    def __init__(self, backend: Any, settings: Settings, mapping: str = "absolute"):
        if mapping not in ("absolute", "relative"):
            raise ValueError("mapping must be absolute or relative")
        if mapping == "absolute" and not backend.supports_absolute:
            raise ValueError("This backend only supports relative mapping")
        self.backend = backend
        self.settings = settings
        self.mapping = mapping
        self.paused = True  # Explicitly enable before sending any mouse events.
        self.button_down = False
        self._pinch_since: float | None = None
        self._release_since: float | None = None
        self._palm_since: float | None = None
        self._palm_armed = True
        self._last_mode = "idle"
        self._last_pointer: tuple[float, float] | None = None
        self._last_pinching = False
        self._cursor_filter = OneEuro2D(settings.min_cutoff, settings.filter_beta)
        self._cursor: tuple[float, float] | None = None
        self._drag_base: tuple[float, float] | None = None
        self._drag_cursor: tuple[float, float] | None = None
        self._scroll_prev: tuple[float, float] | None = None
        self._scroll_x = 0.0
        self._scroll_y = 0.0
        self._last_seen: float | None = None

    def _release_button(self) -> None:
        if self.button_down:
            self.backend.release()
            self.button_down = False
        self._pinch_since = None
        self._release_since = None
        self._drag_base = None
        self._drag_cursor = None

    def _reset_motion(self) -> None:
        self._last_pointer = None
        self._last_pinching = False
        self._cursor = None
        self._cursor_filter.reset()
        self._scroll_prev = None
        self._scroll_x = 0.0
        self._scroll_y = 0.0
        self._drag_base = None
        self._drag_cursor = None
        self._last_mode = "idle"

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._release_button()
        self.paused = paused
        self._reset_motion()

    def toggle_pause(self) -> None:
        self.set_paused(not self.paused)
        self._palm_armed = False  # Require the palm to lower before another toggle.

    def reset(self) -> None:
        """Recalibrate motion after moving the camera / display."""
        self._release_button()
        self._reset_motion()

    def stop(self) -> None:
        """Always release the mouse button on shutdown."""
        self.reset()

    def no_hand(self, now: float) -> Status:
        # On any tracking loss, immediately release a held button to avoid
        # accidentally dragging across the whole desktop.
        self._release_button()
        self._reset_motion()
        self._palm_since = None
        self._palm_armed = True
        return Status("no hand", self.paused, False)

    def process(self, pts: Sequence[Any], now: float, frame_width: int, frame_height: int) -> Status:
        self._last_seen = now
        if len(pts) < 21:
            return self.no_hand(now)

        index, middle, ring, pinky = finger_states(pts)
        ratio = pinch_ratio(pts)
        palm_open = index and middle and ring and pinky

        if palm_open:
            if self._palm_since is None:
                self._palm_since = now
            elapsed = now - self._palm_since
            progress = clamped(elapsed / self.settings.palm_hold_s, 0.0, 1.0)
            if self._palm_armed and elapsed >= self.settings.palm_hold_s:
                self.toggle_pause()
            if self.button_down:
                self._release_button()
            self._reset_motion()
            return Status("hold palm to toggle", self.paused, self.button_down, ratio, progress)
        else:
            self._palm_since = None
            self._palm_armed = True

        if self.paused:
            self._release_button()
            self._reset_motion()
            return Status("paused", True, False, ratio)

        # Once mouse-down occurs, continue drag mode until the pinch releases.
        # This prevents a short finger tracking error from entering scroll mode.
        scrolling = index and middle and not ring and not pinky and not self.button_down and ratio > self.settings.pinch_down_ratio
        if scrolling:
            self._release_button()
            self._cursor_filter.reset()
            self._last_pointer = None
            if self._last_mode != "scroll":
                self._scroll_prev = None
                self._scroll_x = 0.0
                self._scroll_y = 0.0
            self._last_mode = "scroll"
            self._do_scroll(pts, frame_width, frame_height)
            return Status("scroll", False, False, ratio)

        pointer_active = (index and not middle) or ratio <= self.settings.pinch_down_ratio or self.button_down
        if not pointer_active:
            self._release_button()
            self._reset_motion()
            return Status("idle", False, False, ratio)

        if self._last_mode != "pointer":
            self._cursor_filter.reset()
            self._last_pointer = None
        self._last_mode = "pointer"
        self._scroll_prev = None
        self._scroll_x = 0.0
        self._scroll_y = 0.0

        # Enter pinch mode immediately to prevent the fingertip moving toward
        # the thumb from dragging the pointer away from the desired click point.
        pinch_like = ratio <= self.settings.pinch_down_ratio or self.button_down
        self._move_pointer(pts, now, frame_width, frame_height, pinch_like)
        self._update_button(ratio, now)
        return Status("drag" if self.button_down else "pointer", False, self.button_down, ratio)

    def _raw_absolute(self, point: Any) -> tuple[float, float]:
        margin = self.settings.screen_margin
        width = max(self.backend.screen_width - 1, 1)
        height = max(self.backend.screen_height - 1, 1)
        x = clamped((point.x - margin) / (1.0 - 2 * margin), 0.0, 1.0)
        y = clamped((point.y - margin) / (1.0 - 2 * margin), 0.0, 1.0)
        return x * width, y * height

    def _move_pointer(self, pts: Sequence[Any], now: float, fw: int, fh: int, pinching: bool) -> None:
        # Use the index knuckle to track hand movement while pinching, avoiding
        # cursor jumps caused by the fingertip folding toward the thumb.
        point = pts[INDEX_MCP] if pinching else pts[INDEX_TIP]
        if self.mapping == "absolute":
            if pinching:
                if self._drag_base is None:
                    self._drag_base = (point.x, point.y)
                    if self._cursor is None:
                        self._cursor = self._raw_absolute(pts[INDEX_TIP])
                    self._drag_cursor = self._cursor
                    self._cursor_filter.reset()
                x = self._drag_cursor[0] + (point.x - self._drag_base[0]) * self.backend.screen_width / (1 - 2 * self.settings.screen_margin)
                y = self._drag_cursor[1] + (point.y - self._drag_base[1]) * self.backend.screen_height / (1 - 2 * self.settings.screen_margin)
                x = clamped(x, 0.0, self.backend.screen_width - 1)
                y = clamped(y, 0.0, self.backend.screen_height - 1)
            else:
                if self._drag_base is not None:
                    # On leaving drag mode, reset smoothing before following tip.
                    self._cursor_filter.reset()
                self._drag_base = None
                self._drag_cursor = None
                x, y = self._raw_absolute(point)
            x, y = self._cursor_filter.filter(x, y, now)
            self.backend.move_to(x, y)
            self._cursor = (x, y)
        else:
            if pinching != self._last_pinching:
                self._last_pointer = None  # Clutch on mode transition.
                self._cursor_filter.reset()  # Avoid mixing fingertip/knuckle coordinates.
            self._last_pinching = pinching
            self._drag_base = (point.x, point.y) if pinching else None
            current = self._cursor_filter.filter(point.x * fw, point.y * fh, now)
            if self._last_pointer is not None:
                dx = (current[0] - self._last_pointer[0]) * self.settings.cursor_gain
                dy = (current[1] - self._last_pointer[1]) * self.settings.cursor_gain
                self.backend.move_by(dx, dy)
            self._last_pointer = current

    def _update_button(self, ratio: float, now: float) -> None:
        cfg = self.settings
        if not self.button_down:
            if ratio <= cfg.pinch_down_ratio:
                if self._pinch_since is None:
                    self._pinch_since = now
                elif now - self._pinch_since >= cfg.pinch_confirm_s:
                    self.backend.press()
                    self.button_down = True
                    self._release_since = None
            else:
                self._pinch_since = None
        else:
            if ratio >= cfg.pinch_up_ratio:
                if self._release_since is None:
                    self._release_since = now
                elif now - self._release_since >= cfg.pinch_release_s:
                    self._release_button()
            else:
                self._release_since = None

    def _do_scroll(self, pts: Sequence[Any], fw: int, fh: int) -> None:
        mx = (pts[INDEX_TIP].x + pts[MIDDLE_TIP].x) / 2.0
        my = (pts[INDEX_TIP].y + pts[MIDDLE_TIP].y) / 2.0
        if self._scroll_prev is not None:
            dx_px = (mx - self._scroll_prev[0]) * fw
            dy_px = (self._scroll_prev[1] - my) * fh  # Finger up = wheel up.
            if abs(dx_px) < self.settings.scroll_deadzone_px:
                dx_px = 0.0
            if abs(dy_px) < self.settings.scroll_deadzone_px:
                dy_px = 0.0
            scale = self.settings.scroll_gain / self.settings.scroll_pixels_per_tick
            self._scroll_x += dx_px * scale
            self._scroll_y += dy_px * scale
            # Truncation toward zero keeps fractional scroll movement between frames.
            ticks_x, ticks_y = int(self._scroll_x), int(self._scroll_y)
            self._scroll_x -= ticks_x
            self._scroll_y -= ticks_y
            if ticks_x or ticks_y:
                self.backend.scroll(ticks_x, ticks_y)
        self._scroll_prev = (mx, my)
