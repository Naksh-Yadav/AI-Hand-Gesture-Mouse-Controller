import cv2
import logging
import sys
from config import config
from hand_tracker import HandTracker
from hand_role_tracker import HandRoleTracker
from gesture_engine import GestureEngine
from cursor_controller import CursorController
from safety import SafetyManager
from gesture_engine import ClickSequenceRecognizer

# Configure logging
logging.basicConfig(
    level=logging.INFO if config.system['debug_logging'] else logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    print("--- AI Hand Gesture Mouse Controller ---")
    print(f"Control Mode: {config.system['control_mode']}")
    if config.system['control_mode'] == "two_hand":
        print("Instructions (Two-Hand Mode):")
        print("- Hand A (Left-ish): Move Cursor")
        print("- Hand B (Right-ish): Actions")
        print("  - Thumb + Index Pinch: Left Click / Drag")
        print("  - Quick Pinches: Double Click")
        print("  - Index, Middle, Ring Extended: Right Click")
        print("  - Index + Middle Pose + Vertical Move: Scroll")
        print("  - Closed Fist below/above center: Zoom Out / In (release between commands)")
    else:
        print("Instructions (One-Hand Mode):")
        print("- Move Index Finger: Move Cursor")
        print("- Thumb + Index Pinch: Left Click / Drag")
        print("- Index + Middle Pose: Scroll")
        print("- Index, Middle, Ring Extended: Right Click")
    print("- F8: Pause/Resume")
    print("- ESC: Emergency Stop")
    print("-----------------------------------------")

    tracker = HandTracker()
    if not tracker.start():
        logging.error("Failed to start camera. Exiting.")
        sys.exit(1)

    role_tracker = HandRoleTracker()
    action_engine = GestureEngine()
    cursor_engine = GestureEngine()
    controller = CursorController()

    is_mouseDown = False
    click_sequence = ClickSequenceRecognizer(config.gestures['double_click_window'])

    def trigger_pause():
        nonlocal is_mouseDown
        if is_mouseDown:
            controller.release_up()
            is_mouseDown = False
        action_engine.set_paused(not (action_engine.state == GestureEngine.PAUSED))
        cursor_engine.set_paused(not (cursor_engine.state == GestureEngine.PAUSED))
        click_sequence.reset()

    def trigger_stop():
        nonlocal is_mouseDown
        if is_mouseDown:
            controller.release_up()
        tracker.stop()
        cv2.destroyAllWindows()
        logging.info("Application stopped safely.")
        sys.exit(0)

    safety = SafetyManager(controller, trigger_pause, trigger_stop)

    try:
        while True:
            frame = tracker.get_frame()
            if frame is None:
                continue

            landmarks_list, handedness_list = tracker.process_frame(frame)
            if not landmarks_list:
                action_engine.reset_transient(tracking_lost=True)
                click_sequence.reset()
                controller.freeze_cursor(False)
                if is_mouseDown:
                    controller.release_up()
                    is_mouseDown = False
                    logging.info("Tracking lost: released mouse")
                continue

            roles = role_tracker.update(landmarks_list, handedness_list)

            # --- CURSOR LOGIC ---
            cursor_hand_idx = None
            for idx, role in roles.items():
                if role == HandRoleTracker.CURSOR:
                    cursor_hand_idx = idx
                    break

            if (action_engine.state not in (GestureEngine.PAUSED, GestureEngine.ZOOMING)
                    and cursor_hand_idx is not None and cursor_hand_idx < len(landmarks_list)):
                cursor_landmarks = landmarks_list[cursor_hand_idx]
                index_tip = cursor_landmarks.landmark[8]
                screen_x, screen_y = controller.map_coordinates(index_tip.x, index_tip.y)
                controller.move_cursor(screen_x, screen_y)

            # --- ACTION LOGIC ---
            action_hand_idx = None
            for idx, role in roles.items():
                if role == HandRoleTracker.ACTION:
                    action_hand_idx = idx
                    break

            if config.system['control_mode'] == "one_hand":
                action_hand_idx = cursor_hand_idx

            if action_hand_idx is not None and action_hand_idx < len(landmarks_list):
                action_landmarks = landmarks_list[action_hand_idx]
                state = action_engine.update(action_landmarks)

                if state != GestureEngine.PAUSED:
                    # In one-hand mode freeze briefly during an accepted click pose;
                    # in two-hand mode cursor movement comes only from the cursor hand.
                    if (config.system['control_mode'] == 'one_hand'
                            and config.cursor.get('freeze_on_click', False)
                            and state in (GestureEngine.PINCH_PENDING, GestureEngine.PINCH_HELD)):
                        controller.freeze_cursor(True)
                    else:
                        controller.freeze_cursor(False)

                    incompatible = (state in (GestureEngine.DRAGGING,
                                              GestureEngine.SCROLLING,
                                              GestureEngine.ZOOMING)
                                    or action_engine.right_click_started_at is not None)
                    if incompatible:
                        click_sequence.reset()

                    if action_engine.right_click:
                        click_sequence.reset()
                        controller.click(button='right')
                        logging.info("Right click triggered")

                    if state == GestureEngine.DRAGGING:
                        if not is_mouseDown:
                            controller.press_down()
                            is_mouseDown = True
                            logging.info("Drag started")

                    else:
                        if is_mouseDown:
                            controller.release_up()
                            is_mouseDown = False
                            logging.info("Drag released")

                    if action_engine.pinch_completed and not incompatible:
                        click_action = click_sequence.pinch_completed()
                        if click_action == 'double':
                            controller.click(double=True)
                            logging.info("Double click triggered")
                        elif click_action == 'single':
                            controller.click()
                            logging.info("Click triggered")

                    # A pending first pinch becomes a single click only when its
                    # confirmation window expires. Tracking loss/pause cancels it.
                    if state not in (GestureEngine.SCROLLING, GestureEngine.ZOOMING, GestureEngine.DRAGGING):
                        click_action = click_sequence.poll()
                        if click_action == 'single':
                            controller.click()
                            logging.info("Click triggered")

                    if state == GestureEngine.SCROLLING:
                        y_pos = action_landmarks.landmark[8].y
                        center_y = (config.cursor['bounds'][1] + config.cursor['bounds'][3]) / 2
                        delta = (center_y - y_pos) * config.gestures['scroll_speed'] * 10
                        controller.scroll(delta)

                    elif state == GestureEngine.ZOOMING:
                        if action_engine.zoom_direction:
                            controller.zoom(action_engine.zoom_direction)
            else:
                action_engine.reset_transient(tracking_lost=True)
                click_sequence.reset()
                if is_mouseDown:
                    controller.release_up()
                    is_mouseDown = False

            if config.system['preview_enabled']:
                frame = tracker.draw_landmarks(frame, landmarks_list, handedness_list)
                action_state = action_engine.state if 'action_engine' in locals() else "N/A"
                pinch_text = (f"Pinch: {action_engine.pinch_distance:.2f} "
                              f"act={action_engine.pinch_act:.2f} rel={action_engine.pinch_rel:.2f} "
                              f"accepted={action_engine.pinch_accepted} ({action_engine.last_reason})"
                              if action_engine.pinch_distance is not None else action_engine.last_reason)
                status_text = f"Mode: {config.system['control_mode']} | Roles: {roles} | Action: {action_state}"
                cv2.putText(frame, status_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if action_engine.debug:
                    cv2.putText(frame, pinch_text, (10, 58),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
                cv2.imshow("Hand Gesture Mouse Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    trigger_stop()

    except KeyboardInterrupt:
        trigger_stop()
    except Exception as e:
        logging.exception(f"Unexpected error: {e}")
        trigger_stop()

if __name__ == "__main__":
    main()
