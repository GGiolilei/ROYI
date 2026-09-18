import os
import pickle
import time
import cv2


class FaceRecognizer:
    """Recognizes known user identities using OpenCV LBPH Face Recognizer."""

    def __init__(
        self,
        model_path="data/lbph_model.xml",
        label_map_path="data/label_map.pkl",
        cascade_path="data/haarcascade_frontalface_default.xml",
        confidence_threshold=80.0,
    ):
        self.confidence_threshold = confidence_threshold  # Lower is better in LBPH
        self.presence_state = None  # None or string name of active user
        self.last_seen_time = 0
        self.cooldown_seconds = 5  # Reduced cooldown for snappier exit events

        # 1. Load Haar Cascade with fallback paths (Local -> OpenCV Data)
        if os.path.exists(cascade_path):
            active_cascade = cascade_path
        else:
            active_cascade = (
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

        self.face_cascade = cv2.CascadeClassifier(active_cascade)
        if self.face_cascade.empty():
            print(
                f"[ROYI Face Error]: Could not load Haar cascade from {active_cascade}"
            )

        # 2. Initialize OpenCV LBPH Recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_map = {}
        self.is_trained = False

        self.load_model(model_path, label_map_path)

    def load_model(self, model_path, label_map_path):
        """Loads trained LBPH model and ID-to-Name map."""
        if os.path.exists(model_path) and os.path.exists(label_map_path):
            try:
                self.recognizer.read(model_path)
                with open(label_map_path, "rb") as f:
                    self.label_map = pickle.load(f)
                self.is_trained = True
                print(
                    f"[ROYI Face]: Loaded trained model with users: {list(self.label_map.values())}"
                )
            except Exception as e:
                print(f"[ROYI Face Error]: Failed to load model files: {e}")
                self.is_trained = False
        else:
            print(
                "[ROYI Face]: No trained LBPH model found. Running in Unknown-only mode."
            )

    def process(self, rgb_frame):
        """Processes RGB frame, detects face region, predicts identity, and updates presence state."""
        # Convert RGB input frame to Grayscale for OpenCV LBPH
        gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)

        # Detect face bounding boxes
        faces = self.face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        current_time = time.time()
        detected_name = "Unknown"
        event_type = None

        if len(faces) > 0:
            self.last_seen_time = current_time

            # Target the largest face in the frame
            (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
            face_roi = gray_frame[y : y + h, x : x + w]

            # Predict face identity if model is loaded
            if self.is_trained:
                label_id, confidence = self.recognizer.predict(face_roi)

                # LBPH confidence scale: lower value indicates a closer match
                if confidence < self.confidence_threshold:
                    detected_name = self.label_map.get(label_id, "Unknown")

            # Handle state transitions (USER_ENTERED / USER_CHANGED)
            if self.presence_state != detected_name:
                self.presence_state = detected_name
                event_type = "USER_ENTERED"
        else:
            # Handle user leaving frame after cooldown timeout
            if self.presence_state and (
                current_time - self.last_seen_time > self.cooldown_seconds
            ):
                event_type = "USER_LEFT"
                self.presence_state = None

        return event_type, detected_name, faces