import cv2
import numpy as np
import time
from collections import deque
import os
from tensorflow.keras.models import load_model

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "gesture_cnn[1].h5")  # <-- change if your filename differs

# ---------------- Config ----------------
IMG_SIZE = 128
CONF_THRESH = 0.80
SMOOTHING_WINDOW = 7
COOLDOWN = 0.6  # seconds between commands

# Must match your training class order
CLASS_NAMES = [
    "FIST",
    "ONE_FINGER",
    "OPEN_PALM",
    "THUMB_INDEX",
    "THUMB_INDEX_PINKY",
    "THUMB_UP",
    "TWO_FINGER"
]

# ---------------- Load model ----------------
model = load_model(MODEL_PATH)
print("✅ Loaded model:", MODEL_PATH)

# ---------------- Drone simulation (replace with real SDK calls) ----------------
def drone_takeoff(): print("🛫 TAKEOFF")
def drone_land(): print("🛬 LAND")
def drone_emergency(): print("🚨 EMERGENCY STOP")
def drone_exit_emergency(): print("✅ EXIT EMERGENCY")
def drone_up(): print("⬆️ UP")
def drone_down(): print("⬇️ DOWN")
def drone_left(): print("⬅️ LEFT")
def drone_right(): print("➡️ RIGHT")
def drone_up_left(): print("↖️ UP_LEFT")
def drone_up_right(): print("↗️ UP_RIGHT")
def drone_down_left(): print("↙️ DOWN_LEFT")
def drone_down_right(): print("↘️ DOWN_RIGHT")
def drone_rotate_cw(): print("🔁 ROTATE_CW")
def drone_rotate_ccw(): print("🔄 ROTATE_CCW")
def drone_stop(): print("⏸️ STOP")

# ---------------- State machine ----------------
STATE_LANDED = "LANDED"
STATE_FLYING = "FLYING"
STATE_EMERGENCY = "EMERGENCY"
state = STATE_LANDED

last_cmd_time = 0

# ---------------- Temporal smoothing ----------------
history = deque(maxlen=SMOOTHING_WINDOW)

def majority_vote(seq):
    if not seq:
        return None
    return max(set(seq), key=seq.count)

# ---------------- Direction from ROI (FIXED: deadzone + dominance) ----------------
def get_direction_from_roi(gray):
    h, w = gray.shape
    cy, cx = h // 2, w // 2

    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    moments = cv2.moments(thresh)
    if moments["m00"] == 0:
        return None

    x = int(moments["m10"] / moments["m00"])
    y = int(moments["m01"] / moments["m00"])

    dx = x - cx
    dy = y - cy

    DEADZONE = 25      # ignore tiny movements
    DOMINANCE = 1.3    # one axis must clearly beat the other

    if abs(dx) < DEADZONE and abs(dy) < DEADZONE:
        return None

    # Horizontal only if clearly stronger
    if abs(dx) > abs(dy) * DOMINANCE:
        return "RIGHT" if dx > 0 else "LEFT"

    # Vertical only if clearly stronger
    if abs(dy) > abs(dx) * DOMINANCE:
        return "DOWN" if dy > 0 else "UP"

    return None

# ---------------- Diagonal from TWO_FINGER ----------------
def get_diagonal_from_roi(gray):
    h, w = gray.shape
    cy, cx = h // 2, w // 2

    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    moments = cv2.moments(thresh)
    if moments["m00"] == 0:
        return None

    x = int(moments["m10"] / moments["m00"])
    y = int(moments["m01"] / moments["m00"])

    dx = x - cx
    dy = y - cy

    DEADZONE = 25
    DOMINANCE = 1.3

    if abs(dx) < DEADZONE and abs(dy) < DEADZONE:
        return None

    # Decide main axis (same logic as single finger)
    if abs(dx) > abs(dy) * DOMINANCE:
        # Horizontal dominant
        if dx > 0:
            return "UP_RIGHT"     # two-finger RIGHT -> UP_RIGHT
        else:
            return "DOWN_LEFT"    # two-finger LEFT -> DOWN_LEFT

    if abs(dy) > abs(dx) * DOMINANCE:
        # Vertical dominant
        if dy > 0:
            return "DOWN_RIGHT"   # two-finger DOWN -> DOWN_RIGHT
        else:
            return "UP_LEFT"      # two-finger UP -> UP_LEFT

    return None

# ---------------- Execute with safety ----------------
def try_execute(cmd):
    global last_cmd_time, state

    now = time.time()
    if now - last_cmd_time < COOLDOWN:
        return

    if state == STATE_EMERGENCY:
        if cmd == "EXIT_EMERGENCY":
            drone_exit_emergency()
            state = STATE_LANDED
            last_cmd_time = now
        return

    if cmd == "EMERGENCY":
        drone_emergency()
        state = STATE_EMERGENCY
        last_cmd_time = now
        return

    if state == STATE_LANDED:
        if cmd == "TAKEOFF":
            drone_takeoff()
            state = STATE_FLYING
            last_cmd_time = now
        return

    if state == STATE_FLYING:
        if cmd == "LAND":
            drone_land()
            state = STATE_LANDED
        elif cmd == "STOP":
            drone_stop()
        elif cmd == "UP":
            drone_up()
        elif cmd == "DOWN":
            drone_down()
        elif cmd == "LEFT":
            drone_left()
        elif cmd == "RIGHT":
            drone_right()
        elif cmd == "UP_LEFT":
            drone_up_left()
        elif cmd == "UP_RIGHT":
            drone_up_right()
        elif cmd == "DOWN_LEFT":
            drone_down_left()
        elif cmd == "DOWN_RIGHT":
            drone_down_right()
        elif cmd == "ROTATE_CW":
            drone_rotate_cw()
        elif cmd == "ROTATE_CCW":
            drone_rotate_ccw()

        last_cmd_time = now

# ---------------- Command latch (FIX: no repeat until STOP) ----------------
last_executed_command = None
awaiting_stop = False

MOVES = {"UP","DOWN","LEFT","RIGHT","UP_LEFT","UP_RIGHT","DOWN_LEFT","DOWN_RIGHT","ROTATE_CW","ROTATE_CCW"}

def gated_execute(cmd):
    global last_executed_command, awaiting_stop

    if cmd is None:
        return

    # If waiting for STOP, only accept STOP
    if awaiting_stop:
        if cmd == "STOP":
            awaiting_stop = False
            last_executed_command = "STOP"
        return

    # If same command repeated, ignore
    if cmd == last_executed_command:
        return

    # Execute new command
    try_execute(cmd)

    # If it's a movement/rotation, require STOP before repeating
    if cmd in MOVES:
        awaiting_stop = True

    last_executed_command = cmd

# ---------------- Webcam ----------------
cap = cv2.VideoCapture(0)
print("🎥 Running real-time gesture control. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # ROI
    ROI_SIZE = 300
    cx, cy = w // 2, h // 2
    x1 = cx - ROI_SIZE // 2
    y1 = cy - ROI_SIZE // 2
    x2 = cx + ROI_SIZE // 2
    y2 = cy + ROI_SIZE // 2

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    # Draw ROI + center dot
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.circle(frame, (cx, cy), 8, (0,0,255), -1)

    # Prepare for CNN
    img = resized.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)[0]
    conf = float(np.max(preds))
    class_id = int(np.argmax(preds))

    if conf < CONF_THRESH:
        gesture = "UNKNOWN"
    else:
        gesture = CLASS_NAMES[class_id]

    history.append(gesture)
    stable = majority_vote(history)

    command = None

    # Map SHAPE -> ACTION
    if stable == "FIST":
        command = "LAND"
    elif stable == "OPEN_PALM":
        if state == STATE_EMERGENCY:
            command = "EXIT_EMERGENCY"
        else:
            command = "STOP"
    elif stable == "THUMB_UP":
        command = "TAKEOFF"
    elif stable == "THUMB_INDEX":
        command = "ROTATE_CW"
    elif stable == "THUMB_INDEX_PINKY":
        command = "ROTATE_CCW"
    elif stable == "ONE_FINGER":
        direction = get_direction_from_roi(gray)
        command = direction
    elif stable == "TWO_FINGER":
        diag = get_diagonal_from_roi(gray)
        command = diag

    if command is not None:
        gated_execute(command)

    # UI
    cv2.putText(frame, f"STATE: {state}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, f"RAW: {gesture} ({conf:.2f})", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f"STABLE: {stable}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"CMD: {command}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
    cv2.putText(frame, f"LATCH: {'WAIT_STOP' if awaiting_stop else 'READY'}", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,200,0), 2)

    cv2.imshow("CNN Gesture Control", frame)
    cv2.imshow("ROI", resized)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
