import numpy as np
import pyautogui

# Disable PyAutoGUI fail-safe pause for smooth real-time control
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0


class VirtualMouse:

    def __init__(self, smoothing=0.5, margin=100):
        self.screen_w, self.screen_h = pyautogui.size()
        self.smoothing = smoothing  # Higher value = smoother movement
        self.margin = margin  # Boundary padding for easier edge access

        self.prev_x, self.prev_y = self.screen_w // 2, self.screen_h // 2
        self.is_clicking = False

    def move(self, norm_x: float, norm_y: float, frame_w=640, frame_h=480):
        """Maps normalized webcam coordinates (0.0 to 1.0) to screen coordinates with smoothing."""
        # 1. Flip horizontal axis (webcam mirror effect)
        norm_x = 1.0 - norm_x

        # 2. Map coordinates with boundary margin padding
        target_x = np.interp(
            norm_x * frame_w,
            (self.margin, frame_w - self.margin),
            (0, self.screen_w),
        )
        target_y = np.interp(
            norm_y * frame_h,
            (self.margin, frame_h - self.margin),
            (0, self.screen_h),
        )

        # 3. Apply Exponential Moving Average (EMA) for smooth motion
        curr_x = self.prev_x + (target_x - self.prev_x) * (1.0 - self.smoothing)
        curr_y = self.prev_y + (target_y - self.prev_y) * (1.0 - self.smoothing)

        # 4. Move OS Cursor
        pyautogui.moveTo(int(curr_x), int(curr_y))

        self.prev_x, self.prev_y = curr_x, curr_y

    def click(self, button="left"):
        """Performs a single mouse click."""
        pyautogui.click(button=button)

    def press(self):
        """Holds down left mouse button (for drag operations)."""
        if not self.is_clicking:
            pyautogui.mouseDown(button="left")
            self.is_clicking = True

    def release(self):
        """Releases held mouse button."""
        if self.is_clicking:
            pyautogui.mouseUp(button="left")
            self.is_clicking = False