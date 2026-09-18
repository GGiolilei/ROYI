import math

class GestureClassifier:
    @staticmethod
    def _distance(p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def classify(self, landmarks):
        # MediaPipe Landmark IDs: 4=Thumb_Tip, 8=Index_Tip, 12=Middle_Tip, 0=Wrist
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        wrist = landmarks[0]

        # Calculate distances relative to wrist scale
        scale = self._distance(landmarks[0], landmarks[9])  # Wrist to Middle MCP
        pinch_dist = self._distance(thumb_tip, index_tip) / scale
        index_raised = index_tip.y < landmarks[6].y
        middle_raised = middle_tip.y < landmarks[10].y

        if pinch_dist < 0.3:
            return "PINCH"
        elif index_raised and middle_raised:
            return "TWO_FINGERS"
        elif index_raised:
            return "INDEX_UP"
        elif not index_raised and not middle_raised:
            return "FIST"
        
        return "UNKNOWN"