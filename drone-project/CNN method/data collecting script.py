import cv2
import os
import time

# ---------------- Configuration ----------------
DATASET_DIR = "dataset"
IMG_SIZE = 128            # Final image size: 128x128
MAX_IMAGES_PER_CLASS = 1250

# Gesture labels and keys
GESTURES = {
    'w': 'UP',
    's': 'DOWN',
    'a': 'LEFT',
    'd': 'RIGHT',
    'x': 'STOP',
    'q': 'UP_LEFT',
    'e': 'UP_RIGHT',
    'z': 'DOWN_LEFT',
    'c': 'DOWN_RIGHT',
    'r': 'ROTATE_CW',
    't': 'ROTATE_CCW',
    'o': 'TAKEOFF',
    'l': 'LAND',
    'k': 'EMERGENCY',
    'p': 'EXIT_EMERGENCY'
}

# ---------------- Create folders ----------------
os.makedirs(DATASET_DIR, exist_ok=True)

for label in GESTURES.values():
    path = os.path.join(DATASET_DIR, label)
    os.makedirs(path, exist_ok=True)

# ---------------- Count existing images ----------------
def count_images(label):
    path = os.path.join(DATASET_DIR, label)
    return len([f for f in os.listdir(path) if f.endswith(".jpg")])

# ---------------- Webcam ----------------
cap = cv2.VideoCapture(0)

print("=== DATA COLLECTION STARTED ===")
print("Hold the corresponding key to record images.")
print("Press ESC to quit.\n")

for k, v in GESTURES.items():
    print(f"Key '{k}' -> {v}")

# ROI box (center)
ROI_SIZE = 300  # size of crop square

# Key holding state
current_key = None
last_save_time = 0
SAVE_INTERVAL = 0.05  # seconds between saves (~20 fps)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Define ROI in center
    cx, cy = w // 2, h // 2
    x1 = cx - ROI_SIZE // 2
    y1 = cy - ROI_SIZE // 2
    x2 = cx + ROI_SIZE // 2
    y2 = cy + ROI_SIZE // 2

    roi = frame[y1:y2, x1:x2]

    # Draw ROI rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show counters on screen
    y_text = 30
    for label in GESTURES.values():
        cnt = count_images(label)
        cv2.putText(frame, f"{label}: {cnt}", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_text += 18

    # Convert ROI to grayscale and resize
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    # Show preview of what is being saved
    cv2.imshow("ROI (Saved Image Preview)", resized)
    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    # Exit
    if key == 27:  # ESC
        break

    # Check if key corresponds to a gesture
    if chr(key) in GESTURES:
        current_key = chr(key)
    else:
        current_key = None

    # Save images continuously while key is held
    if current_key is not None:
        label = GESTURES[current_key]
        count = count_images(label)

        if count < MAX_IMAGES_PER_CLASS:
            now = time.time()
            if now - last_save_time > SAVE_INTERVAL:
                filename = os.path.join(DATASET_DIR, label, f"{count:05d}.jpg")
                cv2.imwrite(filename, resized)
                last_save_time = now
                print(f"Saved {filename}")
        else:
            cv2.putText(frame, f"{label} DONE (1250 images)", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

cap.release()
cv2.destroyAllWindows()
print("=== DATA COLLECTION FINISHED ===")
