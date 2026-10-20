import logging
import sys
from pynput import keyboard

class SafetyManager:
    def __init__(self, controller_callback, pause_callback, stop_callback):
        self.controller = controller_callback
        self.pause_func = pause_callback
        self.stop_func = stop_callback

        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()

    def _on_press(self, key):
        try:
            if key == keyboard.Key.f8:
                logging.info("Pause/Resume toggled via keyboard")
                self.pause_func()
            elif key == keyboard.Key.esc:
                logging.info("Emergency stop triggered via keyboard")
                self.stop_func()
        except Exception as e:
            logging.error(f"Error in keyboard listener: {e}")

    def stop(self):
        self.listener.stop()

    def release_all_buttons(self):
        # Force release all mouse buttons to prevent stuck state
        import pyautogui
        pyautogui.mouseUp()
