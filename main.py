import threading
import cv2
from core.royi import ROYICore
from vision.camera import Camera
from vision.face import FaceRecognizer
from vision.hand import CompleteHandTracker

stop_event = threading.Event()


def draw_hud(frame, user_name, face_count, active_status):
    """Renders a visual UI overlay on top of the camera feed for real-time feedback."""
    h, w, _ = frame.shape

    # 1. Semi-transparent Top Status Bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 45), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # 2. System Status Text
    cv2.putText(
        frame,
        "ROYI 1.0 VISION ENGINE",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    # 3. Dynamic User Feedback Badge
    if user_name != "Unknown" and face_count > 0:
        status_text = f"ACTIVE USER: {user_name.upper()}"
        badge_color = (0, 255, 0)  # Green for recognized user
    elif face_count > 0:
        status_text = "FACE DETECTED: UNKNOWN"
        badge_color = (0, 165, 255)  # Orange for unknown user
    else:
        status_text = "SEARCHING FOR FACE..."
        badge_color = (200, 200, 200)  # Gray when idle

    cv2.putText(
        frame,
        status_text,
        (w - 280, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        badge_color,
        2,
    )

    # 4. Center Target Reticle (Gives instant visual feedback when positioning face)
    cx, cy = w // 2, h // 2
    reticle_color = badge_color
    cv2.line(frame, (cx - 15, cy), (cx + 15, cy), reticle_color, 1)
    cv2.line(frame, (cx, cy - 15), (cx, cy + 15), reticle_color, 1)


def vision_loop(royi_instance):
    """Runs camera, face recognition, and hand gesture tracking with UI feedback."""
    cam = Camera()
    face_sys = FaceRecognizer()
    hand_sys = CompleteHandTracker()

    print("--- Vision Subsystem Started ---")

    active_user = "Unknown"

    while not stop_event.is_set():
        frame, rgb_frame = cam.read()

        if frame is None:
            print("[ROYI Vision]: Failed to read frame from camera.")
            break

        # 1. Process Face Recognition
        event_type, user_name, faces = face_sys.process(rgb_frame)

        if event_type == "USER_ENTERED":
            active_user = user_name
            print(f"\n[ROYI Vision]: User identified: {user_name}. Welcome back!")
            royi_instance.bus.publish({"type": "USER_ENTERED", "user": user_name})

        elif event_type == "USER_LEFT":
            active_user = "Unknown"
            print("\n[ROYI Vision]: User left camera view.")
            royi_instance.bus.publish({"type": "USER_LEFT"})

        # 2. Draw Target Boxes for Detected Faces
        for x, y, w_box, h_box in faces:
            is_known = user_name != "Unknown"
            color = (0, 255, 0) if is_known else (0, 0, 255)

            # Draw Corner Bounding Box (Tech HUD Style)
            length = 20
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 1)

            # Highlight Corners
            cv2.line(frame, (x, y), (x + length, y), color, 3)
            cv2.line(frame, (x, y), (x, y + length), color, 3)
            cv2.line(frame, (x + w_box, y), (x + w_box - length, y), color, 3)
            cv2.line(frame, (x + w_box, y), (x + w_box, y + length), color, 3)

            # Label Tag above face
            label = f"{user_name}"
            cv2.rectangle(
                frame, (x, y - 25), (x + len(label) * 12, y), color, -1
            )
            cv2.putText(
                frame,
                label,
                (x + 5, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
            )

        # 3. Process Hand Gestures
        hand_sys.process(frame, rgb_frame, royi_instance.bus)

        # 4. Render UI HUD Overlay
        draw_hud(frame, user_name, len(faces), active_user)

        # 5. Display Window
        cv2.imshow("ROYI 1.0 Vision Debug", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    cam.release()
    cv2.destroyAllWindows()
    print("--- Vision Subsystem Stopped ---")


def cli_loop(royi_instance):
    """Handles terminal input."""
    print("--- ROYI 1.0 Core Running ---")
    print(
        "Type 'exit' or 'quit' to stop, or press 'q' in the camera window.\n"
    )

    while not stop_event.is_set():
        try:
            user_input = input("User > ")
            if user_input.lower() in ["exit", "quit"]:
                stop_event.set()
                break
            royi_instance.process_input(user_input)
        except (KeyboardInterrupt, EOFError):
            stop_event.set()
            break


if __name__ == "__main__":
    royi = ROYICore()

    vision_thread = threading.Thread(
        target=vision_loop, args=(royi,), daemon=True
    )
    vision_thread.start()

    cli_loop(royi)

    vision_thread.join(timeout=1.0)
    print("ROYI 1.0 shut down cleanly.")