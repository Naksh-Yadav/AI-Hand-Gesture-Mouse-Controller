import numpy as np
import time
from config import config

class HandRoleTracker:
    """
    Assigns and maintains stable roles (CURSOR vs ACTION) for detected hands.
    """
    CURSOR = "CURSOR"
    ACTION = "ACTION"

    def __init__(self):
        self.roles = {} # Maps hand_id (index in multi_hand_landmarks) to role
        self.last_known_positions = {} # Maps role to last seen center position
        self.assignment_stable_since = 0
        self.stability_threshold = 0.5 # seconds before role is considered stable

    def _get_hand_center(self, landmarks):
        # Simple center of landmarks as a proxy for hand position
        xs = [l.x for l in landmarks.landmark]
        ys = [l.y for l in landmarks.landmark]
        return np.array([np.mean(xs), np.mean(ys)])

    def update(self, multi_hand_landmarks, multi_handedness):
        if not multi_hand_landmarks:
            self.roles = {}
            self.last_known_positions = {}
            return self.roles

        current_hands = []
        for idx, landmarks in enumerate(multi_hand_landmarks):
            center = self._get_hand_center(landmarks)
            current_hands.append({
                'id': idx,
                'center': center,
                'landmarks': landmarks
            })

        if len(current_hands) == 2:
            # Strategy: The hand on the left is CURSOR, the hand on the right is ACTION.
            # We evaluate this every frame to ensure stability even if MediaPipe swaps indices.

            # Find which index is further left
            if current_hands[0]['center'][0] < current_hands[1]['center'][0]:
                self.roles[0] = self.CURSOR
                self.roles[1] = self.ACTION
            else:
                self.roles[0] = self.ACTION
                self.roles[1] = self.CURSOR

        elif len(current_hands) == 1:
            # Single hand: default to CURSOR for backward compatibility/one-hand mode
            self.roles = {0: self.CURSOR}
        else:
            self.roles = {}

        return self.roles
