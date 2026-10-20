import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from cursor_controller import CursorController
from gesture_engine import ClickSequenceRecognizer, GestureEngine
from hand_role_tracker import HandRoleTracker

class MockLandmark:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

class MockHandLandmarks:
    def __init__(self, landmarks_list):
        self.landmark = landmarks_list

def make_hand(pinch_distance=0.08, wrist_y=0.6):
    points = [MockLandmark(0.5, 0.5) for _ in range(21)]
    points[0] = MockLandmark(0.5, wrist_y)
    points[5] = MockLandmark(0.4, 0.5)
    points[17] = MockLandmark(0.6, 0.5)
    points[4] = MockLandmark(0.48, 0.5)
    points[8] = MockLandmark(0.48 + pinch_distance, 0.5)
    # Extended-finger default avoids matching the closed-fist pose.
    for idx, x in ((12, 0.8), (16, 0.9), (20, 1.0)):
        points[idx] = MockLandmark(x, 0.1)
    return MockHandLandmarks(points)

class TestTwoHandControl(unittest.TestCase):
    def setUp(self):
        self.controller = CursorController()
        self.role_tracker = HandRoleTracker()
        self.engine = GestureEngine()

    def test_role_assignment_stability(self):
        # Mock two hands: Hand 0 on left, Hand 1 on right
        h0_landmarks = MockHandLandmarks([MockLandmark(0.2, 0.5) for _ in range(21)])
        h1_landmarks = MockHandLandmarks([MockLandmark(0.8, 0.5) for _ in range(21)])

        # MediaPipe formatted output
        multi_landmarks = [h0_landmarks, h1_landmarks]
        multi_handedness = [MagicMock(), MagicMock()] # Handedness not strictly used in current spatial logic

        roles = self.role_tracker.update(multi_landmarks, multi_handedness)

        self.assertEqual(roles[0], HandRoleTracker.CURSOR)
        self.assertEqual(roles[1], HandRoleTracker.ACTION)

        # Swap input order: Hand 1 on left, Hand 0 on right
        # (Simulating MediaPipe returning them in different order)
        multi_landmarks_swapped = [h1_landmarks, h0_landmarks]
        roles_swapped = self.role_tracker.update(multi_landmarks_swapped, multi_handedness)

        # The one on the left should still be CURSOR
        # Since h1_landmarks is now at index 0 and is at x=0.8 (right), it should be ACTION
        # Since h0_landmarks is now at index 1 and is at x=0.2 (left), it should be CURSOR
        self.assertEqual(roles_swapped[0], HandRoleTracker.ACTION)
        self.assertEqual(roles_swapped[1], HandRoleTracker.CURSOR)

    def test_cursor_independence(self):
        # Ensure action hand movements don't change cursor if we only move action hand
        # This is a logic check in main.py, but we can verify controller behavior
        initial_pos = (100, 100)
        with patch('pyautogui.moveTo') as mock_move:
            # Simulate movement of action hand only (Hand 1)
            # Main loop logic: cursor only moves if cursor_hand_idx is found and processed
            # Here we just verify that calling move_cursor with same coords does nothing new
            # or that we can freeze it.
            self.controller.freeze_cursor(True)
            self.controller.move_cursor(500, 500)
            # moveTo should not have been called because it's frozen
            mock_move.assert_not_called()

    def test_single_click_no_cursor_shift(self):
        # In two-hand mode, the action hand triggers the click.
        # We verify that we can trigger a click without calling move_cursor.
        with patch('pyautogui.click') as mock_click:
            # simulate action hand state = MOVING and was_pinching = True
            self.controller.click()
            mock_click.assert_called_once()

    def test_drag_and_drop_logic(self):
        # Verify press_down and release_up
        with patch('pyautogui.mouseDown') as mock_down, patch('pyautogui.mouseUp') as mock_up:
            self.controller.press_down()
            mock_down.assert_called_once()
            self.controller.release_up()
            mock_up.assert_called_once()

    def test_normalized_pinch_is_scale_independent(self):
        engine = GestureEngine()
        small = make_hand(0.04)
        large = make_hand(0.04)
        # Double all hand coordinates around the origin to simulate image scale.
        large.landmark = [MockLandmark(p.x * 2, p.y * 2, p.z * 2) for p in large.landmark]
        self.assertAlmostEqual(engine.normalized_pinch_distance(small),
                               engine.normalized_pinch_distance(large))

    def test_hysteresis_stability_and_one_completion(self):
        engine = GestureEngine()
        pinch = make_hand(0.04)
        self.assertEqual(engine.update(pinch, now=1.0), GestureEngine.PINCH_PENDING)
        self.assertEqual(engine.update(pinch, now=1.1), GestureEngine.PINCH_HELD)
        # Distance between activation and release remains pinched.
        mid = make_hand(0.08)
        self.assertEqual(engine.update(mid, now=1.15), GestureEngine.PINCH_HELD)
        release = make_hand(0.12)
        engine.update(release, now=1.2)
        self.assertTrue(engine.pinch_completed)
        engine.update(release, now=1.3)
        self.assertFalse(engine.pinch_completed)

    def test_tracking_loss_cancels_unreleased_pinch(self):
        engine = GestureEngine()
        pinch = make_hand(0.04)
        engine.update(pinch, now=1.0)
        engine.update(pinch, now=1.1)
        self.assertEqual(engine.update(None, now=1.12), GestureEngine.TRACKING_LOST)
        engine.update(make_hand(0.12), now=1.2)
        self.assertFalse(engine.pinch_completed)

    def test_brief_noisy_pinch_is_rejected(self):
        engine = GestureEngine()
        engine.update(make_hand(0.04), now=1.0)
        engine.update(make_hand(0.12), now=1.04)
        self.assertFalse(engine.pinch_completed)
        self.assertIn('rejected', engine.last_reason)

    def test_double_click_confirmation_sequences(self):
        recognizer = ClickSequenceRecognizer(window=0.4)
        self.assertIsNone(recognizer.pinch_completed(now=1.0))
        self.assertEqual(recognizer.pinch_completed(now=1.25), 'double')
        self.assertIsNone(recognizer.poll(now=1.3))
        self.assertIsNone(recognizer.pinch_completed(now=2.0))
        self.assertEqual(recognizer.poll(now=2.41), 'single')
        self.assertIsNone(recognizer.pinch_completed(now=3.0))
        self.assertEqual(recognizer.pinch_completed(now=3.5), 'single')
        self.assertEqual(recognizer.poll(now=3.91), 'single')

    def test_tracking_loss_cancels_pending_double_click(self):
        recognizer = ClickSequenceRecognizer(window=0.4)
        recognizer.pinch_completed(now=1.0)
        recognizer.reset()
        self.assertIsNone(recognizer.pinch_completed(now=1.5))
        self.assertIsNone(recognizer.poll(now=1.6))

    def test_zoom_is_edge_triggered_and_directional(self):
        engine = GestureEngine()
        fist = make_hand(0.12, wrist_y=0.3)
        for idx in (8, 12, 16, 20):
            fist.landmark[idx] = MockLandmark(0.5, 0.32)
        engine.update(fist, now=1.0)
        self.assertEqual(engine.zoom_direction, 'out')
        engine.update(fist, now=1.2)
        self.assertIsNone(engine.zoom_direction)
        engine.update(make_hand(0.12), now=1.3)  # release pose
        engine.update(fist, now=1.4)  # cooldown blocks a fresh held pose
        self.assertIsNone(engine.zoom_direction)
        engine.update(make_hand(0.12), now=1.5)
        fist.landmark[0] = MockLandmark(0.5, 0.7)
        for idx in (8, 12, 16, 20):
            fist.landmark[idx] = MockLandmark(0.5, 0.72)
        engine.update(fist, now=2.0)
        self.assertEqual(engine.zoom_direction, 'in')

    def test_right_click_pose_is_stable_and_emits_once(self):
        engine = GestureEngine()
        hand = make_hand(0.12)
        hand.landmark[8] = MockLandmark(0.2, 0.1)
        hand.landmark[12] = MockLandmark(0.5, 0.1)
        hand.landmark[16] = MockLandmark(0.8, 0.1)
        engine.update(hand, now=1.0)
        self.assertFalse(engine.right_click)
        engine.update(hand, now=1.21)
        self.assertTrue(engine.right_click)
        engine.update(hand, now=1.3)
        self.assertFalse(engine.right_click)

    def test_zoom_shortcuts_are_discrete_ctrl_key_events(self):
        with patch('pyautogui.hotkey') as mock_hotkey:
            self.controller.zoom('in')
            self.controller.zoom('out')
        self.assertEqual(mock_hotkey.call_args_list, [
            unittest.mock.call('ctrl', '='), unittest.mock.call('ctrl', '-')
        ])

    def test_click_recognizer_can_be_cancelled_for_pause_or_tracking_loss(self):
        recognizer = ClickSequenceRecognizer(window=0.4)
        recognizer.pinch_completed(now=1.0)
        recognizer.reset()
        self.assertIsNone(recognizer.poll(now=2.0))

    def test_paused_engine_emits_no_gesture(self):
        engine = GestureEngine()
        engine.set_paused(True)
        self.assertEqual(engine.update(make_hand(0.04), now=1.0), GestureEngine.PAUSED)
        self.assertFalse(engine.pinch_completed)
        self.assertIsNone(engine.zoom_direction)

if __name__ == "__main__":
    unittest.main()
