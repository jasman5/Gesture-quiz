import cv2
import os
import time

GESTURES = ['A', 'B', 'C', 'D', 'None']

SAVE_DIR = 'data/gestures/alldata'
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_IMAGES = 200            
DELAY_BETWEEN_SHOTS = 0.1   
BOX_SIZE = 200              

def collect_gesture(label):
    gesture_dir = os.path.join(SAVE_DIR, label)
    os.makedirs(gesture_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print(f"\n🖐 Ready to collect gesture: '{label}'")
    print("Press 's' to START collecting, 'q' to QUIT this gesture.\n")

    started = False
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        x1, y1, x2, y2 = w//2 - BOX_SIZE, h//2 - BOX_SIZE, w//2 + BOX_SIZE, h//2 + BOX_SIZE
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if not started:
            cv2.putText(frame, "Press 's' to start collecting",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(frame, f"Collecting {label}: {count}/{NUM_IMAGES}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            roi = frame[y1+10:y2-10, x1+10:x2-10]
            img_path = os.path.join(gesture_dir, f"{count}.jpg")
            cv2.imwrite(img_path, roi)
            count += 1
            time.sleep(DELAY_BETWEEN_SHOTS)

        cv2.imshow("Gesture Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            started = True
            print("Started capturing...")
        elif key == ord('q') or count >= NUM_IMAGES:
            print(f"Collected {count} images for '{label}'")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    for gesture in GESTURES:
        collect_gesture(gesture)
    print("\n🎉 All gesture data collected successfully!")
