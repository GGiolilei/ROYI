import math
import os
import cv2
import mediapipe as mp
import numpy as np
import pyautogui


class CompleteHandTracker:

    def __init__(self, model_path="data/hand_landmarker.task"):
        self.enabled = True
        self.screen_w, self.screen_h = pyautogui.size()
        self.prev_x, self.prev_y = self.screen_w // 2, self.screen_h // 2
        self.smoothing = 0.5
        self.margin = 60

        # Define MediaPipe Hand Skeleton Bone Connections
        self.HAND_BONES = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),  # Thumb
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),  # Index
            (5, 9),
            (9, 10),
            (10, 11),
            (11, 12),  # Middle
            (9, 13),
            (13, 14),
            (14, 15),
            (15, 16),  # Ring
            (13, 17),
            (17, 18),
            (18, 19),
            (19, 20),
            (0, 17),  # Pinky & Palm
        ]

        if not os.path.exists(model_path):
            print(f"[HandTracker Error]: Missing task model at {model_path}")
            self.detector = None
            return

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
        )
        self.detector = HandLandmarker.create_from_options(options)

    def _calculate_hand_rotation(self, wrist, index_mcp, pinky_mcp):
        """Calculates roll angle (degrees) using wrist and palm landmarks."""
        # Vector from wrist to palm base
        dx = pinky_mcp.x - index_mcp.x
        dy = pinky_mcp.y - index_mcp.y
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        return angle_deg

    def _get_distance(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def process(self, frame, rgb_frame, event_bus):
        if not self.enabled or self.detector is None:
            return

        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)

        if not results.hand_landmarks:
            return

        landmarks = results.hand_landmarks[0]
        if rgb_frame is None:
            rgb_frame = frame
            event_bus = rgb_frame
            
        if not self.enabled or self.detector is None:
            return

        # 1. Draw Full 21-Joint Skeleton & Bones
        points = {}
        for idx, lm in enumerate(landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            points[idx] = (cx, cy)
            # Draw Joint Nodes
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

        for p1, p2 in self.HAND_BONES:
            cv2.line(frame, points[p1], points[p2], (255, 0, 128), 2)

        # 2. Key Landmarks
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]

        # 3. Mouse Movement (Index Finger Tip)
        target_x = np.interp(
            index_tip.x * w, (self.margin, w - self.margin), (0, self.screen_w)
        )
        target_y = np.interp(
            index_tip.y * h, (self.margin, h - self.margin), (0, self.screen_h)
        )

        curr_x = self.prev_x + (target_x - self.prev_x) * (1.0 - self.smoothing)
        curr_y = self.prev_y + (target_y - self.prev_y) * (1.0 - self.smoothing)
        pyautogui.moveTo(int(curr_x), int(curr_y))
        self.prev_x, self.prev_y = curr_x, curr_y

        # 4. Pinch Gesture Detection
        pinch_dist = self._get_distance(thumb_tip, index_tip)
        is_pinched = pinch_dist < 0.045

        if is_pinched:
            cv2.circle(
                frame, points[8], 10, (0, 255, 0), -1
            )  # Green visual indicator on pinch
            pyautogui.mouseDown()
        else:
            pyautogui.mouseUp()

        # 5. Hand Rotation for Volume Control
        rotation_angle = self._calculate_hand_rotation(
            wrist, index_mcp, pinky_mcp
        )

        # Map rotation angle (-45 deg to 45 deg) to volume adjustments
        if rotation_angle > 25:
            pyautogui.press("volumeup")
            vol_status = "VOL UP"
        elif rotation_angle < -25:
            pyautogui.press("volumedown")
            vol_status = "VOL DOWN"
        else:
            vol_status = "NEUTRAL"

        # 6. Render Rotation HUD Feedback
        cv2.putText(
            frame,
            f"ROTATION: {int(rotation_angle)} deg ({vol_status})",
            (15, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )

        # 7. Publish to Event Bus
        if event_bus:
            event_bus.publish({
                "type": "GESTURE_UPDATE",
                "pinch": is_pinched,
                "rotation": rotation_angle,
            })