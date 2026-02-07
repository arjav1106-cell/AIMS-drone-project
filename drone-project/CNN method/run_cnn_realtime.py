import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque, Counter

# ---------------- Configuration ----------------
MODEL_PATH = "CNN method/gesture_cnn.h5"

IMG_SIZE = 128
ROI_SIZE = 300

# Confidence & stability
CONF_THRESHOLD = 0.75     # Only accept predictions above this confidence
SMOOTHING_WINDOW = 7      # Number of recent frames to vote over
STABLE_FRAMES = 5         # How many votes needed for a stable decision

# IMPORTANT: Must match the class order printed during training (train_gen.class_indices)
# Example order you showed:
CLASS_NAMES = [
    'DOWN',
    'DOWN_LEFT',
    'DOWN_RIGHT',
    'EMERGENCY',
    'EXIT_EMERGENCY',
    'LAND',
    'LEFT',
    'RIGHT',
    'ROTATE_CCW',
    'ROTATE_CW',
    'STOP',
    'TAKEOFF',
    'UP',
    'UP_LEFT',
    'UP_RIGHT'
]

# ---------------- Load Model ----------------
model = load_model(MODEL_PATH)
print("✅ Model loaded:", MODEL_PATH)

# ---------------- Webcam ----------------
cap = cv2.VideoCapture(0)

# Queue for temporal smoothing
recent_preds = deque(maxlen=SMOOTHING_WINDOW)

def get_stable_label(pred_queue):
    """Return the most common label if it appears at least STABLE_FRAMES times."""
    if len(pred_queue) < STABLE_FRAMES:
        return None
    counts = Counter(pred_queue)
    label, cnt = counts.most_common(1)[0]
    if cnt >= STABLE_FRAMES:
        return label
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Define ROI in center (same as data collection)
    cx, cy = w // 2, h // 2
    x1 = cx - ROI_SIZE // 2
    y1 = cy - ROI_SIZE // 2
    x2 = cx + ROI_SIZE // 2
    y2 = cy + ROI_SIZE // 2

    # Clamp to frame bounds (safety)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    roi = frame[y1:y2, x1:x2]

    # Draw ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Preprocess ROI: grayscale -> resize -> normalize
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    norm = resized.astype("float32") / 255.0

    # Shape to (1, 128, 128, 1)
    input_tensor = np.expand_dims(norm, axis=(0, -1))

    # Predict
    preds = model.predict(input_tensor, verbose=0)[0]
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))
    raw_label = CLASS_NAMES[class_id]

    # Apply confidence threshold
    if confidence >= CONF_THRESHOLD:
        recent_preds.append(raw_label)
    else:
        recent_preds.append("STOP")  # fallback

    stable_label = get_stable_label(recent_preds)
    display_label = stable_label if stable_label is not None else "STOP"

    # UI
    cv2.putText(frame, f"RAW: {raw_label} ({confidence*100:.1f}%)",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"STABLE: {display_label}",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("CNN Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
