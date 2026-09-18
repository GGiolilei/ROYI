import os
import pickle
import cv2
import numpy as np


def train_opencv_model():
    if not hasattr(cv2, "face"):
        print("[CRITICAL ERROR] cv2.face module not found!")
        return

    dataset_dir = os.path.join("data", "faces")
    if not os.path.exists(dataset_dir):
        print(f"[Error] Directory '{dataset_dir}' does not exist.")
        return

    # 1. Load Local XML Cascade File First
    local_xml = os.path.join("data", "haarcascade_frontalface_default.xml")
    if os.path.exists(local_xml):
        cascade_path = local_xml
    else:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"[Error] Could not load Haar cascade from: {cascade_path}")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_map = {}
    current_label = 0

    print("--- Training OpenCV LBPH Face Recognizer ---")

    for user_name in os.listdir(dataset_dir):
        user_folder = os.path.join(dataset_dir, user_name)
        if not os.path.isdir(user_folder):
            continue

        label_map[current_label] = user_name
        print(f"Processing dataset for: {user_name} (ID: {current_label})")

        for img_name in os.listdir(user_folder):
            img_path = os.path.join(user_folder, img_name)
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None or gray.size == 0:
                continue

            detected = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4
            )
            for x, y, w, h in detected:
                faces.append(gray[y : y + h, x : x + w])
                labels.append(current_label)

        current_label += 1

    if not faces:
        print("[Error] No valid face regions found in the dataset images!")
        return

    # Save outputs
    recognizer.train(faces, np.array(labels))
    recognizer.save("data/lbph_model.xml")

    with open("data/label_map.pkl", "wb") as f:
        pickle.dump(label_map, f)

    print("\n[SUCCESS] Training complete!")
    print(f"Model saved to 'data/lbph_model.xml' using {len(faces)} samples.")


if __name__ == "__main__":
    train_opencv_model()