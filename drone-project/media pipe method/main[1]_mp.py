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

# STOP behavior
STOP_REPEAT_FOR_HOLD = 3   # how many STOP edges trigger HOLD
HOLD_DURATION = 3.0        # seconds to lock/hold position

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

def drone_rotate_cw():
    print("🔁 Drone ROTATE CLOCKWISE")

def drone_rotate_ccw():
    print("🔄 Drone ROTATE ANTI-CLOCKWISE")

def drone_stop():
    print("⏸️ Drone HOVER / STOP")

def drone_emergency():
    print("🚨 EMERGENCY STOP !!!")

# ---------------- State Machine ----------------
STATE_LANDED = "LANDED"
STATE_FLYING = "FLYING"
STATE_EMERGENCY = "EMERGENCY"
drone_state = STATE_LANDED

# ---------------- Command dispatcher ----------------
def execute_command(command):
    global drone_state

    if command == "EMERGENCY":
        drone_emergency()
        drone_state = STATE_EMERGENCY
        return

    if command == "EXIT_EMERGENCY":
        print("✅ EXITED EMERGENCY. Back to LANDED state.")
        drone_state = STATE_LANDED
        return

    if drone_state == STATE_LANDED:
        if command == "TAKEOFF":
            drone_takeoff()
            drone_state = STATE_FLYING
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
        elif command == "ROTATE_CW":
            drone_rotate_cw()
        elif command == "ROTATE_CCW":
            drone_rotate_ccw()
        elif command == "STOP":
            drone_stop()

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

# ---------------- Rotate gesture detection (distance patterns) ----------------
def rotate_gesture(hand_landmarks, w, h):
    wrist = hand_landmarks.landmark[0]
    wx, wy = int(wrist.x * w), int(wrist.y * h)

    tips_idx = {"thumb":4, "index":8, "middle":12, "ring":16, "pinky":20}
    d = {}
    for name, idx in tips_idx.items():
        tip = hand_landmarks.landmark[idx]
        tx, ty = int(tip.x * w), int(tip.y * h)
        d[name] = math.hypot(tx - wx, ty - wy)

    OPEN_T = 120
    CLOSED_T = 60

    thumb_open = d["thumb"] > OPEN_T
    index_open = d["index"] > OPEN_T
    middle_closed = d["middle"] < CLOSED_T
    ring_closed = d["ring"] < CLOSED_T
    pinky_open = d["pinky"] > OPEN_T
    pinky_closed = d["pinky"] < CLOSED_T

    # CW: thumb + index open, others closed
    if thumb_open and index_open and middle_closed and ring_closed and pinky_closed:
        return "ROTATE_CW"

    # CCW: thumb + index + pinky open, middle & ring closed
    if thumb_open and index_open and pinky_open and middle_closed and ring_closed:
        return "ROTATE_CCW"

    return None

# ---------------- Smoothing buffers ----------------
index_buf = deque(maxlen=SMOOTH_WINDOW)
middle_buf = deque(maxlen=SMOOTH_WINDOW)

def smooth_point(buf, x, y):
    buf.append((x, y))
    sx = sum(p[0] for p in buf) / len(buf)
    sy = sum(p[1] for p in buf) / len(buf)
    return int(sx), int(sy)

# ---------------- Edge-trigger + HOLD state ----------------
last_executed_command = None
stop_edge_count = 0
hold_until = 0.0

def should_ignore_due_to_hold(cmd):
    now = time.time()
    if now < hold_until:
        if cmd in {"UP","DOWN","LEFT","RIGHT","UP_RIGHT","UP_LEFT","DOWN_RIGHT","DOWN_LEFT","ROTATE_CW","ROTATE_CCW"}:
            return True
    return False

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

        # --------- EMERGENCY / EXIT EMERGENCY: two hands ----------
        if result.multi_hand_landmarks and len(result.multi_hand_landmarks) >= 2:
            states = []
            for hl in result.multi_hand_landmarks[:2]:
                states.append(hand_open_or_fist(hl, w, h))
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            # Enter emergency: both fists
            if states[0] == "FIST" and states[1] == "FIST":
                detected_command = "EMERGENCY"
            # Exit emergency: both open palms (only if currently in emergency)
            elif states[0] == "OPEN" and states[1] == "OPEN" and drone_state == STATE_EMERGENCY:
                detected_command = "EXIT_EMERGENCY"

        # If not emergency-related, process single-hand logic
        if result.multi_hand_landmarks and detected_command not in {"EMERGENCY", "EXIT_EMERGENCY"}:
            hand_landmarks = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # If in EMERGENCY, ignore all other gestures
            if drone_state == STATE_EMERGENCY:
                detected_command = "STOP"
            else:
                # First: TAKEOFF / LAND
                hand_state = hand_open_or_fist(hand_landmarks, w, h)
                if hand_state == "OPEN":
                    detected_command = "TAKEOFF"
                elif hand_state == "FIST":
                    detected_command = "LAND"
                else:
                    # Then: ROTATION gestures
                    rot = rotate_gesture(hand_landmarks, w, h)
                    if rot is not None:
                        detected_command = rot
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

        stable_command, hold_time, required_time = get_stable_command_time_based(detected_command)

        # ---------------- Edge-trigger + STOP/HOLD handling ----------------
        now = time.time()

        # Emergency: immediate, override everything
        if stable_command == "EMERGENCY" and last_executed_command != "EMERGENCY":
            execute_command("EMERGENCY")
            last_executed_command = "EMERGENCY"
            stop_edge_count = 0
            hold_until = 0.0

        # Exit emergency: immediate
        elif stable_command == "EXIT_EMERGENCY" and last_executed_command != "EXIT_EMERGENCY":
            execute_command("EXIT_EMERGENCY")
            last_executed_command = "EXIT_EMERGENCY"
            stop_edge_count = 0
            hold_until = 0.0
            # Reset gesture timing to avoid accidental immediate commands
            last_detected_command = None
            gesture_start_time = None

        # Fast STOP: execute immediately on edge
        elif stable_command == "STOP" and last_executed_command != "STOP":
            stop_edge_count += 1
            if stop_edge_count >= STOP_REPEAT_FOR_HOLD:
                hold_until = now + HOLD_DURATION
                print(f"🧲 HOLD engaged for {HOLD_DURATION:.1f}s")
                stop_edge_count = 0
            execute_command("STOP")
            last_executed_command = "STOP"

        # Other commands: edge-triggered, and respect HOLD
        elif stable_command and stable_command != last_executed_command:
            if stable_command != "STOP":
                stop_edge_count = 0

            if not should_ignore_due_to_hold(stable_command):
                execute_command(stable_command)
                last_executed_command = stable_command

        # UI
        hold_active = time.time() < hold_until
        cv2.putText(frame, f"STATE: {drone_state}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"DETECTED: {detected_command}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"HOLD: {'ON' if hold_active else 'OFF'}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255) if hold_active else (255, 255, 255), 2)
        cv2.putText(frame, f"GESTURE HOLD: {hold_time:.1f}s / {required_time:.1f}s", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        cv2.imshow("Gesture Controlled Drone - Emergency + Exit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
