import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

click_time = 0
scroll_time = 0
snap_time = 0
home_time = 0
prev_x, prev_y = 0, 0
smoothening = 7
cursor_boost = 1.5  # Increase cursor reach

def fingers_status(hand_landmarks):
    fingers = []
    tips = [8, 12, 16, 20]
    dips = [6, 10, 14, 18]
    for tip, dip in zip(tips, dips):
        fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y else 0)
    return fingers

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            # Boosted & smoothed cursor
            adj_x = np.interp(x, [0, w], [0, screen_width * cursor_boost])
            adj_y = np.interp(y, [0, h], [0, screen_height * cursor_boost])
            adj_x = np.clip(adj_x, 0, screen_width)
            adj_y = np.clip(adj_y, 0, screen_height)

            curr_x = prev_x + (adj_x - prev_x) / smoothening
            curr_y = prev_y + (adj_y - prev_y) / smoothening
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            fingers = fingers_status(hand_landmarks)

            # 1️⃣ Pinch click (thumb & index)
            click_distance = math.hypot(
                int(thumb_tip.x * w) - x,
                int(thumb_tip.y * h) - y
            )
            if click_distance < 40 and time.time() - click_time > 0.5:
                pyautogui.click()
                click_time = time.time()
                cv2.circle(frame, (x, y), 15, (0, 255, 255), cv2.FILLED)

            # 2️⃣ Peace to scroll down
            if fingers == [1, 1, 0, 0] and time.time() - scroll_time > 1:
                pyautogui.scroll(-800)
                scroll_time = time.time()
                print("📜 Scrolling down")

            # 3️⃣ Open palm = Go to desktop/home (5 fingers up)
            if fingers == [1, 1, 1, 1] and hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
                if time.time() - home_time > 2:
                    pyautogui.hotkey('win', 'd')
                    home_time = time.time()
                    print("🏠 Show Desktop")

            # 4️⃣ Snap gesture = Screenshot (thumb & middle finger close)
            snap_dist = math.hypot(
                int(thumb_tip.x * w) - int(middle_tip.x * w),
                int(thumb_tip.y * h) - int(middle_tip.y * h)
            )
            if snap_dist < 35 and time.time() - snap_time > 2:
                pyautogui.screenshot("screenshot.png")
                print("📸 Screenshot Taken")
                snap_time = time.time()

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
