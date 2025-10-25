import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import screen_brightness_control as sbc
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ---------- CONFIG ----------
SMOOTHING = 0.2
CLICK_DIST_THRESHOLD = 0.05
SCROLL_SENSITIVITY = 80
DEBUG = True
CLICK_COOLDOWN = 0.4
# ----------------------------

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# Screen size
screen_w, screen_h = pyautogui.size()

# Webcam
cap = cv2.VideoCapture(0)
cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

# States
prev_x, prev_y = 0, 0
last_left_click_time = 0
last_right_click_time = 0
scroll_mode = False
scroll_anchor_y, scroll_anchor_x = 0, 0

# Init volume control (safe)
volume = None
try:
    devices = AudioUtilities.GetSpeakers()
    # NOTE: use iid (pycaw exposes GUID this way)
    interface = devices.Activate(IAudioEndpointVolume.iid, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
except Exception as e:
    # If pycaw fails (non-Windows or missing dependency), continue without volume control
    print("Warning: volume control unavailable:", e)
    volume = None

def normalized_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def finger_up(lm, tip, pip):
    """Check if finger is up: y_tip < y_pip (remember Mediapipe y increases downward)"""
    return lm[tip][1] < lm[pip][1]

print("Starting Virtual Mouse. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read from camera. Exiting.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm = [(lm_obj.x, lm_obj.y) for lm_obj in hand_landmarks.landmark]

        # Palm center (choose a set of palm landmarks and average)
        palm_indices = [0, 1, 2, 5, 9, 13, 17]
        palm_pts = np.array([lm[i] for i in palm_indices])
        palm_center = palm_pts.mean(axis=0)

        # Map to screen (clamp to screen size)
        screen_x = np.interp(palm_center[0], (0, 1), (0, screen_w))
        screen_y = np.interp(palm_center[1], (0, 1), (0, screen_h))
        screen_x = max(0, min(screen_w - 1, screen_x))
        screen_y = max(0, min(screen_h - 1, screen_y))

        # Smooth movement
        cur_x = prev_x + (screen_x - prev_x) * (1 - SMOOTHING)
        cur_y = prev_y + (screen_y - prev_y) * (1 - SMOOTHING)

        if not scroll_mode:
            try:
                pyautogui.moveTo(cur_x, cur_y, duration=0)
            except Exception:
                # fail-safe in case moveTo raises
                pass
        prev_x, prev_y = cur_x, cur_y

        # Finger tips (x,y)
        thumb_tip = lm[4]
        index_tip = lm[8]
        middle_tip = lm[12]

        # Distances
        dist_thumb_index = normalized_distance(thumb_tip, index_tip)
        now = time.time()

        # FINGER STATES
        index_up = finger_up(lm, 8, 6)
        middle_up = finger_up(lm, 12, 10)
        thumb_up = finger_up(lm, 4, 3)

        # --- LEFT CLICK (Thumb + Index pinch short) ---
        if dist_thumb_index < CLICK_DIST_THRESHOLD and not scroll_mode:
            if (now - last_left_click_time) > CLICK_COOLDOWN:
                pyautogui.click(button='left')
                last_left_click_time = now
                cv2.putText(frame, "Left Click", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # --- RIGHT CLICK (Index + Middle up, thumb down) ---
        if index_up and middle_up and not thumb_up:
            if (now - last_right_click_time) > CLICK_COOLDOWN:
                pyautogui.click(button='right')
                last_right_click_time = now
                cv2.putText(frame, "Right Click", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # --- SCROLL MODE (Thumb + Index pinch hold) ---
        # Enter scroll mode when pinch is present; exit when released
        if dist_thumb_index < CLICK_DIST_THRESHOLD:
            if not scroll_mode:
                scroll_mode = True
                scroll_anchor_y = palm_center[1]
                scroll_anchor_x = palm_center[0]
        else:
            scroll_mode = False

        if scroll_mode:
            dy = palm_center[1] - scroll_anchor_y
            dx = palm_center[0] - scroll_anchor_x
            if abs(dy) > 0.005:  # small deadzone
                try:
                    pyautogui.scroll(int(-dy * SCROLL_SENSITIVITY))
                except Exception:
                    pass
            if abs(dx) > 0.005:
                try:
                    pyautogui.hscroll(int(-dx * SCROLL_SENSITIVITY))
                except Exception:
                    pass
            scroll_anchor_y = palm_center[1]
            scroll_anchor_x = palm_center[0]

        # BRIGHTNESS CONTROL (Index up only)
        if index_up and not thumb_up and not middle_up:
            dy = palm_center[1] - 0.5
            if dy < -0.05:
                try:
                    sbc.set_brightness("+5")
                except Exception:
                    pass
                cv2.putText(frame, "Brightness UP", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            elif dy > 0.05:
                try:
                    sbc.set_brightness("-5")
                except Exception:
                    pass
                cv2.putText(frame, "Brightness DOWN", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # VOLUME CONTROL (Thumb up only) - works only if pycaw initialized
        if thumb_up and not index_up and not middle_up and volume is not None:
            dy = palm_center[1] - 0.5
            try:
                current_vol = volume.GetMasterVolumeLevelScalar()
            except Exception:
                current_vol = None
            if current_vol is not None:
                if dy < -0.05:
                    volume.SetMasterVolumeLevelScalar(min(current_vol + 0.05, 1.0), None)
                    cv2.putText(frame, "Volume UP", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                elif dy > 0.05:
                    volume.SetMasterVolumeLevelScalar(max(current_vol - 0.05, 0.0), None)
                    cv2.putText(frame, "Volume DOWN", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                cv2.putText(frame, "Volume ctrl unavailable", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # DEBUG DISPLAY
        if DEBUG:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cx, cy = int(palm_center[0]*cam_w), int(palm_center[1]*cam_h)
            cv2.circle(frame, (cx, cy), 8, (0,255,0), -1)
            cv2.putText(frame, f"Scroll:{scroll_mode}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.imshow("Virtual Mouse - Press 'q' to exit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()