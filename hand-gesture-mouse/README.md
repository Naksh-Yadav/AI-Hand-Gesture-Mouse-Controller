# AI Hand Gesture Mouse Controller
A real-time, webcam-based mouse controller for Windows using MediaPipe and OpenCV.

## Features
- **Cursor Movement**: Control the mouse with your index finger.
- **Left Click**: Hold a thumb/index pinch for the stability interval, then release.
- **Double Click**: Complete a second pinch within the configured confirmation window. A single click waits for that window to expire.
- **Drag & Drop**: Hold the pinch gesture.
- **Scrolling**: Pose with index and middle fingers extended.
- **Right Click**: Pose with index, middle, and ring fingers extended.
- **Safety**: Emergency stop (ESC) and Pause/Resume (F8).
- **Zoom**: Make a closed fist above the frame center for zoom out or below center for zoom in. In two-hand mode use the action hand; in one-hand mode the same hand controls both, and pointer movement pauses while the fist pose is active. Release between commands.

## Installation
1. Ensure you have Python 3.11+ installed.
2. Clone or navigate to the project directory.
3. Create a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Launching
Run the application:
```bash
python main.py
```

## Keyboard Shortcuts
- `F8`: Pause/Resume gesture control.
- `ESC`: Emergency stop and exit.
- `F9`: Toggle webcam preview (if implemented).

## Configuration
Edit `config.toml` to adjust:
- Camera index and resolution.
- Cursor sensitivity and smoothing.
- Normalized pinch activation/release thresholds, click stability, double-click timing, drag timing, zoom cooldown, and scroll speed.
- Screen mapping bounds.

Pinch distance is the 3D thumb-tip/index-tip distance divided by the index-MCP/pinky-MCP span. Activation and release use separate thresholds to prevent chatter. Accepted short pinches click on release; a single-click is delayed by `double_click_window` so a second pinch can produce one double-click without an extra first single-click. Pinches held through `drag_hold_duration` drag instead.

Zoom sends Ctrl+= (zoom in) or Ctrl+- (zoom out) through PyAutoGUI. Support depends on the target application; some apps use different shortcuts. `zoom_cooldown` limits repeated commands, and the pose must be released before it can trigger again. Debug overlay details are shown when `system.debug_logging = true`.

## Known Limitations
- Accuracy depends on lighting and webcam quality; physical webcam behavior still needs validation on the target setup.
- The two-hand role assignment uses left/right screen position. Zoom and other actions run on the assigned action hand; pointer movement runs only on the cursor hand.
- Ctrl+= and Ctrl+- are common zoom shortcuts, but application support varies.
