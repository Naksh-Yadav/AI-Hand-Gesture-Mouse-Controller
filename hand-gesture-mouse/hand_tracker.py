import cv2
import mediapipe as mp
import numpy as np
import logging
from config import config

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2, # Increased to support two-hand control
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.cam_index = config.camera['index']
        self.width = config.camera['width']
        self.height = config.camera['height']
        self.mirrored = config.camera['mirrored']

        self.cap = None
        self.current_frame = None

    def start(self):
        self.cap = cv2.VideoCapture(self.cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.cap.isOpened():
            logging.error(f"Could not open camera {self.cam_index}")
            return False
        return True

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        if self.mirrored:
            frame = cv2.flip(frame, 1)

        self.current_frame = frame
        return frame

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        # Return both landmarks and handedness for role assignment
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks, results.multi_handedness
        return None, None

    def stop(self):
        if self.cap:
            self.cap.release()

    def draw_landmarks(self, frame, landmarks_list, handedness_list=None):
        if landmarks_list:
            for idx, landmarks in enumerate(landmarks_list):
                label = handedness_list[idx].classification[0].label if handedness_list else f"Hand {idx}"
                self.mp_draw.draw_landmarks(
                    frame,
                    landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS
                )
                # Draw label near the wrist
                wrist = landmarks.landmark[0]
                h, w, _ = frame.shape
                cv2.putText(frame, label, (int(wrist.x * w), int(wrist.y * h) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame
