import pyautogui
import numpy as np
from config import config
from smoothing import OneEuroFilter

class CursorController:
    def __init__(self):
        pyautogui.FAILSAFE = True

        self.screen_w, self.screen_h = pyautogui.size()

        self.filter_x = OneEuroFilter(freq=30, mincutoff=1.0, beta=0.01)
        self.filter_y = OneEuroFilter(freq=30, mincutoff=1.0, beta=0.01)

        self.bounds = config.cursor['bounds'] # [x_min, y_min, x_max, y_max]
        self.sensitivity = config.cursor['sensitivity']

        self.frozen_pos = None

    def map_coordinates(self, x, y):
        x_mapped = (x - self.bounds[0]) / (self.bounds[2] - self.bounds[0])
        y_mapped = (y - self.bounds[1]) / (self.bounds[3] - self.bounds[1])

        x_mapped = np.clip(x_mapped, 0, 1)
        y_mapped = np.clip(y_mapped, 0, 1)

        screen_x = x_mapped * self.screen_w
        screen_y = y_mapped * self.screen_h

        return screen_x, screen_y

    def move_cursor(self, x, y):
        if self.frozen_pos:
            return

        smooth_x = self.filter_x.filter(x)
        smooth_y = self.filter_y.filter(y)

        pyautogui.moveTo(smooth_x, smooth_y, _pause=False)

    def freeze_cursor(self, freeze=True):
        if freeze:
            self.frozen_pos = (pyautogui.position())
        else:
            self.frozen_pos = None

    def click(self, button='left', double=False):
        if double:
            pyautogui.doubleClick()
        elif button == 'left':
            pyautogui.click()
        elif button == 'right':
            pyautogui.rightClick()

    def press_down(self, button='left'):
        if button == 'left':
            pyautogui.mouseDown()
        elif button == 'right':
            pyautogui.mouseDown(button='right')

    def release_up(self, button='left'):
        if button == 'left':
            pyautogui.mouseUp()
        elif button == 'right':
            pyautogui.mouseUp(button='right')

    def scroll(self, amount):
        pyautogui.scroll(int(amount))

    def zoom(self, direction):
        # direction: 'in' or 'out'
        # Use standard keys: Ctrl + '=' is Zoom In, Ctrl + '-' is Zoom Out
        if direction == 'in':
            pyautogui.hotkey('ctrl', '=')
        else:
            pyautogui.hotkey('ctrl', '-')
