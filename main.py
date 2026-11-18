#!/usr/bin/env python3
"""Air Mouse: webcam-based pointer, pinch-click/drag, and two-finger scroll.

Quick start:
    python -m pip install -r requirements.txt
    python main.py

Starts PAUSED. Hold your open palm for 0.9 s to enable it.
Press Q/Esc in the preview window to quit, P to pause, R to recalibrate.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys
import time
from urllib.request import urlopen

from airmouse.backend import create_backend
from airmouse.gestures import GestureEngine, Settings, Status

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL = Path(__file__).parent / "models" / "hand_landmarker.task"

# Skeleton pairs based on MediaPipe's 21-landmark hand schema.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def download_model(path: Path) -> None:
    """Download Google's official on-device hand model once on first run."""
    if path.is_file() and path.stat().st_size > 100_000:
        return
    print(f"Downloading MediaPipe Hand Landmarker model to {path} ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    try:
        with urlopen(MODEL_URL, timeout=35) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(128 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                if output.tell() > 50_000_000:
                    raise RuntimeError("Model download exceeded safety size limit")
        if temporary.stat().st_size < 100_000:
            raise RuntimeError("Downloaded model was unexpectedly small")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print("Model downloaded.")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--backend", choices=["auto", "pynput", "uinput"], default="auto")
    parser.add_argument("--mapping", choices=["absolute", "relative"], default=None,
                        help="Default: absolute for pynput, relative for uinput")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--sensitivity", type=float, default=2.4,
                        help="Relative pointer movement multiplier")
    parser.add_argument("--scroll-speed", type=float, default=1.0)
    parser.add_argument("--pinch-down", type=float, default=0.30,
                        help="Palm-width-normalized pinch activation threshold")
    parser.add_argument("--pinch-up", type=float, default=0.48,
                        help="Palm-width-normalized pinch release threshold")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Path for .task model (downloaded automatically if absent)")
    return parser


def draw_hand(cv2, image, landmarks, state: Status) -> None:
    height, width = image.shape[:2]
    coords = [(int(p.x * width), int(p.y * height)) for p in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(image, coords[a], coords[b], (90, 210, 255), 2, cv2.LINE_AA)
    for x, y in coords:
        cv2.circle(image, (x, y), 4, (255, 235, 135), -1, cv2.LINE_AA)
    cv2.circle(image, coords[8], 9,
               (50, 180, 60) if state.button_down else (40, 220, 240),
               2, cv2.LINE_AA)
    if state.pinch_ratio is not None:
        cv2.line(image, coords[4], coords[8], (30, 160, 250), 2, cv2.LINE_AA)


def draw_hud(cv2, image, status: Status, fps: float, mapping: str, backend: str) -> None:
    width = image.shape[1]
    cv2.rectangle(image, (0, 0), (width, 103), (25, 25, 25), -1)
    color = (60, 180, 255) if status.paused else (80, 220, 90)
    title = "PAUSED" if status.paused else "ACTIVE"
    cv2.putText(image, f"AIR MOUSE  [{title}]  {status.gesture.upper()}", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    cv2.putText(image, f"{backend} | {mapping} | {fps:.0f} FPS", (12, 51),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (225, 225, 225), 1, cv2.LINE_AA)
    cv2.putText(image, "Palm hold: toggle | P: pause | R: reset | Q/Esc: quit", (12, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (200, 200, 200), 1, cv2.LINE_AA)
    if status.pause_progress > 0:
        end_x = int(12 + (width - 24) * status.pause_progress)
        cv2.rectangle(image, (12, 90), (end_x, 97), (50, 210, 190), -1)


def run(args: argparse.Namespace) -> int:
    # Import heavier native libraries only after CLI parsing.
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Run: python -m pip install -r requirements.txt", file=sys.stderr)
        return 2

    if args.pinch_up <= args.pinch_down:
        print("--pinch-up must be larger than --pinch-down", file=sys.stderr)
        return 2
    if args.width < 160 or args.height < 120 or args.scroll_speed <= 0 or args.sensitivity <= 0:
        print("Invalid frame size or sensitivity", file=sys.stderr)
        return 2

    try:
        download_model(args.model)
    except Exception as exc:
        print(f"Cannot get hand model: {exc}\n"
              f"Download this file manually and run with --model PATH:\n{MODEL_URL}", file=sys.stderr)
        return 2

    try:
        mouse = create_backend(args.backend)
    except Exception as exc:
        print(f"Mouse backend unavailable: {exc}", file=sys.stderr)
        return 2

    mapping = args.mapping or ("absolute" if mouse.supports_absolute else "relative")
    if mapping == "absolute" and not mouse.supports_absolute:
        print("This mouse backend cannot use absolute mapping; choose --mapping relative.", file=sys.stderr)
        mouse.close()
        return 2

    settings = Settings(
        cursor_gain=args.sensitivity,
        scroll_gain=args.scroll_speed,
        pinch_down_ratio=args.pinch_down,
        pinch_up_ratio=args.pinch_up,
    )
    engine = GestureEngine(mouse, settings, mapping)
    cap = None
    window_name = "Air Mouse - camera preview"

    try:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # May be ignored by some drivers.
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open camera {args.camera}; check Camera permissions")

        BaseOptions = mp.tasks.BaseOptions
        Vision = mp.tasks.vision
        options = Vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(args.model)),
            running_mode=Vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.60,
            min_hand_presence_confidence=0.60,
            min_tracking_confidence=0.55,
        )
        print("Air Mouse ready. It starts PAUSED for safety.")
        print("Hold OPEN PALM for ~0.9 s or press P with preview focused to enable.")
        print("Index: pointer | Thumb + index: click/drag | Index + middle: scroll.")
        print("Press Q/Esc in the preview to quit. Move focus to preview for hotkeys.")

        last_time = time.monotonic()
        last_timestamp_ms = -1
        fps_samples: deque[float] = deque(maxlen=20)
        missing_count = 0
        status = Status("paused", True, False)
        with Vision.HandLandmarker.create_from_options(options) as model:
            while True:
                success, frame = cap.read()
                if not success:
                    missing_count += 1
                    engine.no_hand(time.monotonic())
                    if missing_count > 15:
                        raise RuntimeError("Camera stopped returning frames")
                    continue
                missing_count = 0
                now = time.monotonic()
                dt = max(now - last_time, 1e-5)
                fps_samples.append(1.0 / dt)
                last_time = now
                fps = sum(fps_samples) / len(fps_samples)

                frame = cv2.flip(frame, 1)  # Natural mirror: move right to move right.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = np.ascontiguousarray(rgb)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = max(last_timestamp_ms + 1, int(now * 1000))
                last_timestamp_ms = timestamp_ms
                result = model.detect_for_video(mp_image, timestamp_ms)

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    status = engine.process(landmarks, now, frame.shape[1], frame.shape[0])
                    draw_hand(cv2, frame, landmarks, status)
                else:
                    status = engine.no_hand(now)

                draw_hud(cv2, frame, status, fps, mapping, type(mouse).__name__)
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
                if key == ord("p"):
                    engine.toggle_pause()
                if key == ord("r"):
                    engine.reset()

        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Air Mouse stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.stop()  # Never leave left mouse button held down.
        if cap is not None:
            cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        mouse.close()


if __name__ == "__main__":
    raise SystemExit(run(create_parser().parse_args()))
