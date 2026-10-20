"""Time-aware gesture recognition. This module never dispatches desktop input."""
import math
import time

from config import config


class ClickSequenceRecognizer:
    """Converts completed pinch pulses into single/double click actions.

    A first click is held for ``window`` seconds so a second deliberate pinch
    can become one double-click without also sending an extra single-click.
    """

    def __init__(self, window=0.4):
        self.window = max(0.0, float(window))
        self.pending_at = None

    def pinch_completed(self, now=None):
        now = time.monotonic() if now is None else now
        if self.pending_at is not None and now - self.pending_at <= self.window:
            self.pending_at = None
            return "double"
        action = "single" if self.pending_at is not None else None
        self.pending_at = now
        return action

    def poll(self, now=None):
        now = time.monotonic() if now is None else now
        if self.pending_at is not None and now - self.pending_at > self.window:
            self.pending_at = None
            return "single"
        return None

    def reset(self):
        self.pending_at = None


class GestureEngine:
    INACTIVE = "INACTIVE"
    READY = "READY"
    MOVING = "MOVING"
    PINCH_PENDING = "PINCH_PENDING"
    PINCH_HELD = "PINCH_HELD"
    DRAGGING = "DRAGGING"
    SCROLLING = "SCROLLING"
    ZOOMING = "ZOOMING"
    PAUSED = "PAUSED"
    TRACKING_LOST = "TRACKING_LOST"

    def __init__(self):
        gestures = config.gestures
        self.state = self.INACTIVE
        self.pinch_act = float(gestures['pinch_activation_threshold'])
        self.pinch_rel = float(gestures['pinch_release_threshold'])
        if self.pinch_rel <= self.pinch_act:
            raise ValueError("pinch_release_threshold must be greater than pinch_activation_threshold")
        self.pinch_stability = float(gestures.get('click_stability_duration', 0.08))
        self.right_click_stability = float(gestures.get('right_click_stability_duration', 0.2))
        self.drag_hold = float(gestures.get('drag_hold_duration', 0.3))
        self.scroll_act = float(gestures.get('scroll_activation_threshold', 0.06))
        self.zoom_cooldown = float(gestures.get('zoom_cooldown', 0.6))
        self.debug = bool(config.system.get('debug_logging', False))
        self.pinch_started_at = None
        self.last_zoom_at = float('-inf')
        self.was_zoom_pose = False
        self.pinch_distance = None
        self.pinch_accepted = False
        self.last_reason = "waiting for landmarks"
        self.zoom_direction = None
        self.pinch_completed = False
        self.right_click = False
        self.right_click_started_at = None
        self.right_click_latched = False

    @staticmethod
    def calculate_distance(p1, p2):
        return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)

    def normalized_pinch_distance(self, landmarks):
        try:
            points = landmarks.landmark
            pinch = self.calculate_distance(points[4], points[8])
            # MCP span provides a palm-size reference, independent of image scale.
            palm = self.calculate_distance(points[5], points[17])
            if not math.isfinite(pinch) or not math.isfinite(palm) or palm < 1e-6:
                return None
            return pinch / palm
        except (AttributeError, IndexError, TypeError, ValueError):
            return None

    def reset_transient(self, tracking_lost=False):
        self.pinch_started_at = None
        self.pinch_accepted = False
        self.pinch_completed = False
        self.right_click = False
        self.right_click_started_at = None
        self.right_click_latched = False
        self.was_zoom_pose = False
        self.zoom_direction = None
        self.last_reason = "tracking lost" if tracking_lost else "reset"
        if self.state != self.PAUSED:
            self.state = self.TRACKING_LOST if tracking_lost else self.MOVING

    def update(self, landmarks, now=None):
        now = time.monotonic() if now is None else now
        self.pinch_completed = False
        self.right_click = False
        self.zoom_direction = None
        if self.state == self.PAUSED:
            self.last_reason = "paused"
            return self.state
        if landmarks is None:
            self.reset_transient(tracking_lost=True)
            return self.state

        points = landmarks.landmark
        if len(points) < 21:
            self.reset_transient(tracking_lost=True)
            self.last_reason = "incomplete landmarks"
            return self.state
        self.pinch_distance = self.normalized_pinch_distance(landmarks)
        if self.pinch_distance is None:
            self.reset_transient(tracking_lost=True)
            self.last_reason = "invalid or degenerate landmarks"
            return self.state

        pinch_candidate = (self.pinch_distance < self.pinch_rel if self.pinch_started_at is not None
                           else self.pinch_distance < self.pinch_act)
        if pinch_candidate:
            if self.pinch_started_at is None:
                self.pinch_started_at = now
                self.pinch_accepted = False
            elapsed = now - self.pinch_started_at
            if elapsed >= self.pinch_stability:
                self.pinch_accepted = True
            self.last_reason = "stable pinch" if self.pinch_accepted else "pinch awaiting stability"
            if self.pinch_accepted:
                self.state = self.DRAGGING if elapsed >= self.drag_hold else self.PINCH_HELD
            else:
                self.state = self.PINCH_PENDING
            return self.state

        if self.pinch_started_at is not None:
            accepted = self.pinch_accepted
            elapsed = now - self.pinch_started_at
            self.pinch_started_at = None
            self.pinch_accepted = False
            if accepted and elapsed < self.drag_hold:
                self.pinch_completed = True
                self.last_reason = "accepted pinch released"
            else:
                self.last_reason = "brief/noisy pinch rejected" if not accepted else "drag released"

        # A closed fist is a deliberate zoom pose. One event per pose entry;
        # release is required before another event can be generated.
        wrist = points[0]
        fist = all(self.calculate_distance(points[i], wrist) < 0.25 for i in (8, 12, 16, 20))
        if fist:
            self.state = self.ZOOMING
            if not self.was_zoom_pose and now - self.last_zoom_at >= self.zoom_cooldown:
                self.zoom_direction = 'out' if wrist.y < 0.5 else 'in'
                self.last_zoom_at = now
                self.last_reason = f"zoom {self.zoom_direction} accepted"
            else:
                self.last_reason = "zoom held or cooling down"
            self.was_zoom_pose = True
            return self.state
        self.was_zoom_pose = False

        # Three separated extended fingertips are the existing right-click pose.
        index_middle = self.calculate_distance(points[8], points[12])
        middle_ring = self.calculate_distance(points[12], points[16])
        right_pose = (self.pinch_distance > self.pinch_rel
                      and index_middle > self.scroll_act * 2
                      and middle_ring > self.scroll_act * 2)
        if right_pose:
            if self.right_click_started_at is None:
                self.right_click_started_at = now
            if (not self.right_click_latched
                    and now - self.right_click_started_at >= self.right_click_stability):
                self.right_click = True
                self.right_click_latched = True
                self.state = self.READY
                self.last_reason = "right click accepted"
                return self.state
        else:
            self.right_click_started_at = None
            self.right_click_latched = False

        thumb_middle = self.calculate_distance(points[4], points[12])
        if (self.pinch_distance > self.pinch_rel and thumb_middle > self.scroll_act
                and index_middle < self.scroll_act * 2):
            self.state = self.SCROLLING
            self.last_reason = "scroll pose"
        else:
            self.state = self.MOVING
            if not self.pinch_completed and "rejected" not in self.last_reason:
                self.last_reason = "no action pose"
        return self.state

    def set_paused(self, paused):
        if paused:
            self.reset_transient()
            self.state = self.PAUSED
        elif self.state == self.PAUSED:
            self.state = self.MOVING

