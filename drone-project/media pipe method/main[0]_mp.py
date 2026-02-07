import cv2
import mediapipe as mp
import time
import math
from collections import deque

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ---------------- Tunables ----------------
CENTER_DEADZONE = 80     # pixels: center "STOP" radius
TWO_FINGER_DIST = 40     # pixels: below this => two-finger (diagonal) mode
STABLE_TIME_MOVE = 0.3
STABLE_TIME_CRITICAL = 3.0

# Extra margin to declare "strongly vertical" in two-finger mode
VERT_PRIORITY_MARGIN = 1.2  # multiply CENTER_DEADZONE by this

# Smoothing window (number of frames)
SMOOTH_WINDOW = 5

# ---------------- Drone control abstraction (SIMULATED) ----------------
def drone_takeoff():
    print("🛫 Drone TAKEOFF")

def drone_land():
    print("🛬 Drone LAND")

def drone_up():
    print("⬆️ Drone moving UP")

def drone_down():
    print("⬇️ Drone moving DOWN")

def drone_left():
    print("⬅️ Drone moving LEFT")

def drone_right():
    print("➡️ Drone moving RIGHT")

def drone_up_right():
    print("↗️ Drone moving UP-RIGHT")

def drone_up_left():
    print("↖️ Drone moving UP-LEFT")

def drone_down_right():
    print("↘️ Drone moving DOWN-RIGHT")

def drone_down_left():
    print("↙️ Drone moving DOWN-LEFT")

def drone_stop():
    print("⏸️ Drone HOVER / STOP")

# ---------------- State Machine ----------------
STATE_LANDED = "LANDED"
STATE_FLYING = "FLYING"
drone_state = STATE_LANDED

# ---------------- Command dispatcher with cooldown ----------------
last_cmd_time = 0
CMD_DELAY = 0.5

def execute_command(command):
    global last_cmd_time, drone_state

    now = time.time()
    if now - last_cmd_time < CMD_DELAY:
        return

    if drone_state == STATE_LANDED:
        if command == "TAKEOFF":
            drone_takeoff()
            drone_state = STATE_FLYING
        last_cmd_time = now
        return

    if drone_state == STATE_FLYING:
        if command == "LAND":
            drone_land()
            drone_state = STATE_LANDED
        elif command == "UP":
            drone_up()
        elif command == "DOWN":
            drone_down()
        elif command == "LEFT":
            drone_left()
        elif command == "RIGHT":
            drone_right()
        elif command == "UP_RIGHT":
            drone_up_right()
        elif command == "UP_LEFT":
            drone_up_left()
        elif command == "DOWN_RIGHT":
            drone_down_right()
        elif command == "DOWN_LEFT":
            drone_down_left()
        elif command == "STOP":
            drone_stop()

        last_cmd_time = now

# ---------------- Time-based Gesture Stability ----------------
last_detected_command = None
gesture_start_time = None
CRITICAL_COMMANDS = {"TAKEOFF", "LAND"}

def get_stable_command_time_based(current_command):
    global last_detected_command, gesture_start_time

    now = time.time()

    if current_command != last_detected_command:
        last_detected_command = current_command
        gesture_start_time = now
        return None, 0.0, STABLE_TIME_MOVE

    required_time = STABLE_TIME_CRITICAL if current_command in CRITICAL_COMMANDS else STABLE_TIME_MOVE
    hold_time = now - gesture_start_time if gesture_start_time else 0.0

    if hold_time >= required_time:
        return current_command, hold_time, required_time
    else:
        return None, hold_time, required_time

# ---------------- Open palm / Fist detection (robust) ----------------
def hand_open_or_fist(hand_landmarks, w, h):
    wrist = hand_landmarks.landmark[0]
    wx, wy = int(wrist.x * w), int(wrist.y * h)

    tips = [8, 12, 16, 20]  # index, middle, ring, pinky
    dists = []
    for t in tips:
        tip = hand_landmarks.landmark[t]
        tx, ty = int(tip.x * w), int(tip.y * h)
        d = math.hypot(tx - wx, ty - wy)
        dists.append(d)

    OPEN_THRESH = 120
    FIST_THRESH = 60

    if all(d > OPEN_THRESH for d in dists):
        return "OPEN"
    elif all(d < FIST_THRESH for d in dists):
        return "FIST"
    else:
        return "OTHER"

# ---------------- Smoothing buffers ----------------
index_buf = deque(maxlen=SMOOTH_WINDOW)
middle_buf = deque(maxlen=SMOOTH_WINDOW)

def smooth_point(buf, x, y):
    buf.append((x, y))
    sx = sum(p[0] for p in buf) / len(buf)
    sy = sum(p[1] for p in buf) / len(buf)
    return int(sx), int(sy)

# ---------------- Main Loop ----------------
with mp_hands.Hands(min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Screen center
        cx, cy = w // 2, h // 2
        cv2.circle(frame, (cx, cy), CENTER_DEADZONE, (255, 0, 0), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        detected_command = "STOP"

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # First: robust TAKEOFF / LAND
            hand_state = hand_open_or_fist(hand_landmarks, w, h)
            if hand_state == "OPEN":
                detected_command = "TAKEOFF"
            elif hand_state == "FIST":
                detected_command = "LAND"
            else:
                # Raw fingertip positions
                index_tip = hand_landmarks.landmark[8]
                middle_tip = hand_landmarks.landmark[12]

                ix_raw, iy_raw = int(index_tip.x * w), int(index_tip.y * h)
                mx_raw, my_raw = int(middle_tip.x * w), int(middle_tip.y * h)

                # Smooth them
                ix, iy = smooth_point(index_buf, ix_raw, iy_raw)
                mx, my = smooth_point(middle_buf, mx_raw, my_raw)

                # Draw smoothed tips
                cv2.circle(frame, (ix, iy), 8, (0, 0, 255), -1)
                cv2.circle(frame, (mx, my), 8, (0, 255, 0), -1)

                # Distance between tips decides mode
                dist = math.hypot(ix - mx, iy - my)

                # Use index finger for dead zone / primary direction
                dx = ix - cx
                dy = iy - cy

                if abs(dx) < CENTER_DEADZONE and abs(dy) < CENTER_DEADZONE:
                    detected_command = "STOP"
                else:
                    # TWO-FINGER MODE -> Diagonals with VERTICAL PRIORITY
                    if dist < TWO_FINGER_DIST:
                        # Average point
                        ax = (ix + mx) // 2
                        ay = (iy + my) // 2
                        dx2 = ax - cx
                        dy2 = ay - cy

                        v_strong = CENTER_DEADZONE * VERT_PRIORITY_MARGIN
                        both_below = (iy - cy) > v_strong and (my - cy) > v_strong
                        both_above = (iy - cy) < -v_strong and (my - cy) < -v_strong

                        if both_below:
                            detected_command = "DOWN_RIGHT"
                        elif both_above:
                            detected_command = "UP_LEFT"
                        else:
                            if dx2 < 0 and dy2 < 0:
                                detected_command = "UP_LEFT"
                            elif dx2 > 0 and dy2 < 0:
                                detected_command = "UP_RIGHT"
                            elif dx2 < 0 and dy2 > 0:
                                detected_command = "DOWN_LEFT"
                            elif dx2 > 0 and dy2 > 0:
                                detected_command = "DOWN_RIGHT"
                            else:
                                detected_command = "STOP"
                    # ONE-FINGER MODE -> 4 directions (VERTICAL PRIORITY)
                    else:
                        if dy > CENTER_DEADZONE:
                            detected_command = "DOWN"
                        elif dy < -CENTER_DEADZONE:
                            detected_command = "UP"
                        elif dx > CENTER_DEADZONE:
                            detected_command = "RIGHT"
                        elif dx < -CENTER_DEADZONE:
                            detected_command = "LEFT"
                        else:
                            detected_command = "STOP"
        else:
            detected_command = "STOP"

        stable_command, hold_time, required_time = get_stable_command_time_based(detected_command)

        # UI
        cv2.putText(frame, f"STATE: {drone_state}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"DETECTED: {detected_command}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"HOLD: {hold_time:.1f}s / {required_time:.1f}s", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        if stable_command:
            cv2.putText(frame, f"EXECUTING: {stable_command}", (30, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            execute_command(stable_command)

        cv2.imshow("Gesture Controlled Drone - Smoothed Tips", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
