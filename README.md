# Air Mouse (Python)

A complete webcam-based gesture controller. Written around the **MediaPipe Tasks Hand Landmarker** (21 points), **OpenCV** for capture and preview, and interchangeable OS mouse drivers.

## Gestures

| Gesture | Action |
| --- | --- |
| Index finger extended; middle finger lowered | Move mouse pointer |
| Touch thumb to index fingertip; release | Left click |
| Hold thumb against index fingertip and move your hand | Left-button drag (drop by releasing the pinch) |
| Index and middle fingers extended; ring and pinky curled | Two-dimensional scrolling, up/down and left/right |
| Open palm (all four fingers extended) held for about 0.9 seconds | Toggle controller pause/active |
| Press `P` with preview window focused | Toggle pause |
| Press `R` with preview window focused | Recalibrate/reset pointer state |
| Press `Q` or `Esc` with preview window focused | Quit and release any held mouse button |

**Important:** Starts **PAUSED**. Enable deliberately with an open-palm hold or `P`. The app has an emergency button release on tracking loss and shutdown. Hotkeys are local to the preview window, *not global*.

## Start on macOS / Windows / Linux X11

1. Install 64-bit Python **3.11 or 3.12**. Other current supported Python versions may also work. On macOS, ensure your Python installation includes `tkinter` (check `python -m tkinter`).
2. Open a terminal in this extracted folder.
3. Create a virtual environment and install dependencies:

   **macOS/Linux:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python main.py
   ```

   **Windows PowerShell:**

   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\python.exe -m pip install --upgrade pip
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   .venv\Scripts\python.exe main.py
   ```

On the **first run**, the app downloads Google's official MediaPipe `hand_landmarker.task` model into `models/`. The model executes locally afterward; video frames are not uploaded by this code. If download is blocked, manually download the official model URL (shown on error) and use `--model /path/to/hand_landmarker.task`.

4. Allow camera access. On macOS also grant **Accessibility** to the terminal or app running Python in **System Settings > Privacy & Security**. Restart the terminal if changes don't take effect.
5. A camera preview appears. Hold one open palm for about one second to enable mouse control. Point with one index finger. Use `Q` to quit.

### Example command options

```bash
python main.py --camera 1
python main.py --mapping relative --sensitivity 3.2
python main.py --scroll-speed 1.5
python main.py --pinch-down 0.28 --pinch-up 0.45
python main.py --backend pynput --mapping absolute
python main.py --help
```

### Absolute vs relative mapping

- **Absolute (default on Windows/macOS/Linux X11):** The index fingertip maps from an inner camera rectangle to the **primary screen**. At the edges, the cursor is clamped. Screens with unusual scaling or multi-monitor arrangements may need extra calibration.
- **Relative (default on Linux uinput):** Use hand motion like a trackpad. Move the index fingertip to move the cursor; lower the finger to reposition your hand without moving the pointer. `--sensitivity` adjusts gain.
- Pinching temporarily switches cursor tracking from the index tip to the index knuckle, which prevents the finger-folding motion from creating a large jump. This may still require hand/camera-specific calibration on release.

## Linux Wayland

`pynput` is an **X11-oriented mouse backend** on Linux. A Wayland session usually cannot accept ordinary X11-based global mouse injection. This project includes a separate, optional `evdev` **uinput virtual mouse** backend on Linux. It uses **relative** movements and works only when the compositor accepts the virtual device and you are authorized to use `/dev/uinput`.

```bash
# Ubuntu/Debian examples. These commands configure OS access; review them first.
sudo modprobe uinput
ls -l /dev/uinput
python -m pip install -r requirements-wayland.txt
python main.py --backend uinput --mapping relative
```

**Do not run the whole application using `sudo`.** On many distributions, `/dev/uinput` is root-owned and permission-restricted. Follow your distribution's documented approach for granting your user access (for example, a carefully scoped udev rule or administrator-managed group). Any permission change to `/dev/uinput` permits virtual input injection and should be treated as a security-sensitive decision. Group membership/udev rules and compositor policies differ among systems. Some Wayland setups may not allow this backend; use an X11 session or adapt the backend to your compositor's RemoteDesktop/libei portal.

## Design and reliability

- MediaPipe Tasks `VIDEO` mode: reuses hand tracking between frames.
- Relative palm-width-normalized pinch thresholds (`--pinch-down`, `--pinch-up`).
- Hysteresis and short confirmation intervals for press/release.
- Thumb-index pinch holds a mouse-down until release; dragging works naturally.
- Scrolling uses the **change in midpoint** of index and middle fingertip positions, with a dead zone and fractional wheel-tick accumulation.
- One Euro low-pass smoothing filter reduces hand-jitter and adapts to speed.
- Pause gesture is held and requires open-hand reset between toggles.
- Loss of tracking releases a pressed mouse button; clean exit releases it too.
- 640×480 at 30 FPS requested (camera drivers may pick another format).
- Runs on-device after model download. No video recording or telemetry is implemented.

This is a functional starter, **not a guarantee of production reliability or identical behavior on every laptop**. Test and tune under your lighting, with your camera, display, OS privileges and hand size.

## Troubleshooting

- **Nothing moves:** Controller starts PAUSED. Hold four extended fingers for 0.9 s or click preview window and press `P`.
- **Camera unavailable:** Close Zoom/Teams; check camera permissions; try `--camera 1`.
- **Mouse doesn't move on macOS:** Grant Accessibility permission to the terminal or Python app.
- **Camera window refuses to open:** Run from a regular desktop session with GUI; OpenCV GUI display is required.
- **Linux Wayland injection fails:** Install `requirements-wayland.txt`, check `/dev/uinput` exists and has authorized access, and ensure compositor accepts uinput devices.
- **Tk errors:** Install your system's Python Tk package or use a Python distribution with Tk support. The pynput backend currently initializes Tk even when selecting relative mode.
- **Spurious clicks:** Reduce `--pinch-down` (e.g., 0.26), increase separation between thumb and index, improve illumination.
- **Pinch not detected:** Increase `--pinch-down` slightly (e.g., 0.34); keep palm visible.
- **Cursor too slow:** Relative mapping: increase `--sensitivity`. Absolute mapping: move finger across the active area (inner 74% of the image).
- **Scrolling reversed:** Swap signs inside `GestureEngine._do_scroll` if your preferred natural scrolling direction differs.
- **Horizontal scrolling doesn't work in one app:** Some apps/platforms ignore horizontal wheel events.
- **Multiple monitors:** Current absolute mapping is to the **primary screen only**. Use relative mapping or extend the screen-geometry code.

## Run unit tests

```bash
python -m pip install pytest
python -m pytest -q
```

The tests use a fake mouse and fake hand landmarks, so no webcam is required.

## Code layout

```text
main.py                  Camera capture, on-device model, preview, CLI, clean shutdown
airmouse/backend.py      Pynput and Linux uinput backends
airmouse/gestures.py     Pinch, click/drag, pause, scroll, smoothing and mapping
requirements.txt        Standard platforms
requirements-wayland.txt Linux Wayland extra dependency
tests/                  Synthetic gesture tests
```

## References

- https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
- https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
- https://pynput.readthedocs.io/en/latest/mouse.html
- https://pynput.readthedocs.io/en/latest/limitations.html
- https://python-evdev.readthedocs.io/en/latest/apidoc.html
