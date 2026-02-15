import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import joblib
from collections import deque

MODEL_PATH = "gesture_ann.pt"
LABELS_PATH = "labels.joblib"

CONF_THRESH = 0.6
SMOOTH_WINDOW = 7

# ---------------- Preprocess (MUST MATCH COLLECTION) ----------------
def preprocess_landmarks(lm):
    lm = np.array(lm).reshape(21, 3)
    lm = lm - lm[0]
    max_dist = np.max(np.linalg.norm(lm, axis=1))
    if max_dist > 0:
        lm = lm / max_dist
    return lm.flatten()

# ---------------- Load labels ----------------
le = joblib.load(LABELS_PATH)
classes = le.classes_
num_classes = len(classes)

# ---------------- Model ----------------
class GestureANN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = GestureANN(63, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ---------------- MediaPipe ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

pred_buffer = deque(maxlen=SMOOTH_WINDOW)
last_cmd = None
armed = True

def classify_direction(lm):
    x0, y0 = lm[0], lm[1]
    x8, y8 = lm[8*3], lm[8*3+1]
    dx = x8 - x0
    dy = y8 - y0
    if abs(dx) > abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    else:
        return "DOWN" if dy > 0 else "UP"

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        cv2.circle(frame, (w//2, h//2), 8, (0,255,0), -1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        command = "NONE"

        if res.multi_hand_landmarks:
            hands_lms = res.multi_hand_landmarks

            # Two-hand logic
            if len(hands_lms) == 2:
                preds = []
                for hand_lm in hands_lms:
                    lm = []
                    for i in range(21):
                        lm += [hand_lm.landmark[i].x, hand_lm.landmark[i].y, hand_lm.landmark[i].z]
                    lm = preprocess_landmarks(lm)
                    X = torch.tensor([lm], dtype=torch.float32)
                    with torch.no_grad():
                        out = model(X)
                        pred = classes[torch.argmax(out, dim=1).item()]
                        preds.append(pred)

                if preds[0] == "LAND" and preds[1] == "LAND":
                    command = "EMERGENCY"
                elif preds[0] == "STOP" and preds[1] == "STOP":
                    command = "EXIT_EMERGENCY"

            else:
                hand_lm = hands_lms[0]
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                lm = []
                for i in range(21):
                    lm += [hand_lm.landmark[i].x, hand_lm.landmark[i].y, hand_lm.landmark[i].z]

                lm = preprocess_landmarks(lm)
                X = torch.tensor([lm], dtype=torch.float32)

                with torch.no_grad():
                    out = model(X)
                    prob = torch.softmax(out, dim=1)[0]
                    conf, idx = torch.max(prob, dim=0)

                if conf.item() > CONF_THRESH:
                    shape = classes[idx.item()]
                    command = shape

        pred_buffer.append(command)
        stable_cmd = max(set(pred_buffer), key=pred_buffer.count)

        final_cmd = "NONE"
        if stable_cmd == "STOP":
            armed = True
            final_cmd = "STOP"
        else:
            if armed and stable_cmd != "NONE":
                final_cmd = stable_cmd
                armed = False

        cv2.putText(frame, f"Command: {final_cmd}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.imshow("ANN Gesture Control", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
