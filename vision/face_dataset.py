import cv2
import os
import time

def capture_face_dataset(user_name="Gio", max_samples=40):
    save_path = os.path.join("data", "faces", user_name)
    os.makedirs(save_path, exist_ok=True)

    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    count = 0
    print(f"--- Capturing dataset for {user_name} ---")
    print("Look at the camera. Move your head slowly in different angles.")
    time.sleep(2)

    while count < max_samples:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw bounding box on display
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Save cropped face image
            face_img = frame[y:y+h, x:x+w]
            count += 1
            file_path = os.path.join(save_path, f"{count:03d}.jpg")
            cv2.imwrite(file_path, face_img)
            print(f"Captured {count}/{max_samples}")
            time.sleep(0.1)  # Brief pause between captures

        cv2.putText(
            frame, f"Captured: {count}/{max_samples}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        cv2.imshow("ROYI Face Dataset Collector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Dataset creation complete! Images saved to {save_path}")

if __name__ == "__main__":
    name = input("Enter your name (default 'Gio'): ").strip() or "Gio"
    capture_face_dataset(user_name=name)