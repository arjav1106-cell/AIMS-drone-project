import cv2
import csv
import os
import time
import numpy as np
import mediapipe as mp

# ---------------- Config ----------------
OUTPUT_CSV = "landmarks_dataset.csv"
SAMPLES_PER_CLASS = 3000

CLASSES = [
    "UP", "DOWN", "LEFT", "RIGHT",
    "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT",
    "STOP", "LAND", "TAKEOFF", "ROTATE_CW", "ROTATE_CCW"
]

KEY_MAP = {
    ord('1'): "UP",
    ord('2'): "DOWN",
    ord('3'): "LEFT",
    ord('4'): "RIGHT",
    ord('5'): "UP_LEFT",
    ord('6'): "UP_RIGHT",
    ord('7'): "DOWN_LEFT",
    ord('8'): "DOWN_RIGHT",
    ord('9'): "STOP",
    ord('0'): "LAND",
    ord('q'): "TAKEOFF",
    ord('w'): "ROTATE_CW",
    ord('e'): "ROTATE_CCW",
}

# ---------------- MediaPipe ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ---------------- Preprocess ----------------
def preprocess_landmarks(lm):
    lm = np.array(lm).reshape(21, 3)

    # Translate so wrist is origin
    lm = lm - lm[0]

    # Scale normalization
    max_dist = np.max(np.linalg.norm(lm, axis=1))
    if max_dist > 0:
        lm = lm / max_dist

    return lm.flatten().tolist()

# ---------------- CSV ----------------
file_exists = os.path.exists(OUTPUT_CSV)

counts = {c: 0 for c in CLASSES}

if file_exists:
    import pandas as pd
    df = pd.read_csv(OUTPUT_CSV)
    for c in CLASSES:
        counts[c] = (df["label"] == c).sum()

csv_file = open(OUTPUT_CSV, mode="a", newline="")
writer = csv.writer(csv_file)

if not file_exists:
    header = []
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    header.append("label")
    writer.writerow(header)

# ---------------- Webcam ----------------
cap = cv2.VideoCapture(0)

current_class = None
last_save = 0
SAVE_INTERVAL = 0.03

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    print("Press keys to select class, ESC to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        cv2.circle(frame, (w//2, h//2), 8, (0,255,0), -1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            hand_lm = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            lm = []
            for i in range(21):
                lm += [hand_lm.landmark[i].x, hand_lm.landmark[i].y, hand_lm.landmark[i].z]

            lm = preprocess_landmarks(lm)

            if current_class and counts[current_class] < SAMPLES_PER_CLASS:
                now = time.time()
                if now - last_save > SAVE_INTERVAL:
                    writer.writerow(lm + [current_class])
                    csv_file.flush()
                    counts[current_class] += 1
                    last_save = now

        y = 30
        for c in CLASSES:
            cv2.putText(frame, f"{c}: {counts[c]}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            y += 18

        cv2.putText(frame, f"Current: {current_class}", (10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.imshow("Collect ANN Landmarks", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key in KEY_MAP:
            if counts[KEY_MAP[key]] < SAMPLES_PER_CLASS:
                current_class = KEY_MAP[key]


cap.release()
cv2.destroyAllWindows()
csv_file.close()
print("Done. Saved to", OUTPUT_CSV)
