import cv2
import numpy as np
import mediapipe as mp
# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
canvas = None
brush_thickness = 7

# Finger tip landmarks (thumb to pinky)
finger_tips_ids = [4, 8, 12, 16, 20]

# Default drawing values
draw_color = (255, 0, 0)  # Blue
xp, yp = 0, 0

def count_fingers(hand_landmarks):
    fingers = []

    # Get landmark positions
    lm = hand_landmarks.landmark

    # Thumb (tip > base x for right hand)
    if lm[finger_tips_ids[0]].x < lm[finger_tips_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other 4 fingers
    for i in range(1, 5):
        if lm[finger_tips_ids[i]].y < lm[finger_tips_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm = handLms.landmark
            cx, cy = int(lm[8].x * w), int(lm[8].y * h)  # Index tip

            fingers_up = count_fingers(handLms)
            cv2.putText(frame, f'Fingers: {fingers_up}', (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            # Change color based on finger count
            if fingers_up == 1:
                draw_color = (255, 0, 0)  # Blue
            elif fingers_up == 2:
                draw_color = (0, 255, 0)  # Green
            elif fingers_up == 3:
                draw_color = (0, 0, 255)  # Red
            elif fingers_up == 4:
                draw_color = (0, 255, 255)  # Yellow
            elif fingers_up == 5:
                draw_color = (128, 0, 128)  # Purple
            # Only draw when 1 finger is up
            if fingers_up == 1:
                if xp == 0 and yp == 0:
                    xp, yp = cx, cy
                cv2.line(canvas, (xp, yp), (cx, cy), draw_color, brush_thickness)
                xp, yp = cx, cy
            else:
                xp, yp = 0, 0  # Lift brush

            mp_draw.draw_landmarks(frame, handLms, mp.solutions.hands.HAND_CONNECTIONS)

    else:
        xp, yp = 0, 0

    output = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow("Air Drawing - Color Switch", output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
