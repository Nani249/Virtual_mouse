import cv2
import mediapipe
import numpy as np
import autopy
import pyautogui  # Added for scrolling
import time

cap = cv2.VideoCapture(0)
initHand = mediapipe.solutions.hands
mainHand = initHand.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
draw = mediapipe.solutions.drawing_utils
wScr, hScr = autopy.screen.size()
pX, pY = 0, 0
cX, cY = 0, 0

last_right_click_time = 0  # Timestamp for the last right click
CLICK_COOLDOWN = 1  # 1-second cooldown
scrolling = False  # Flag to control continuous scrolling

def handLandmarks(colorImg):
    landmarkList = []
    landmarkPositions = mainHand.process(colorImg)
    landmarkCheck = landmarkPositions.multi_hand_landmarks

    if landmarkCheck:
        for hand in landmarkCheck:
            for index, landmark in enumerate(hand.landmark):
                draw.draw_landmarks(img, hand, initHand.HAND_CONNECTIONS)
                h, w, c = img.shape
                centerX, centerY = int(landmark.x * w), int(landmark.y * h)
                landmarkList.append([index, centerX, centerY])

    return landmarkList

def fingers(landmarks):
    fingerTips = []
    tipIds = [4, 8, 12, 16, 20]

    for id in tipIds:
        if landmarks[id][2] < landmarks[id - 2][2]:
            fingerTips.append(1)
        else:
            fingerTips.append(0)

    return fingerTips

while True:
    check, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lmList = handLandmarks(imgRGB)

    if len(lmList) != 0:
        finger = fingers(lmList)

        if finger[1] == 1 and finger[2] == 1 and finger[3] == 1 and finger[4] == 0:  # Three fingers
            scrolling = True
            pyautogui.scroll(5)  # Continuous scroll up using pyautogui

        elif finger[1] == 1 and finger[2] == 1 and finger[3] == 1 and finger[4] == 1:  # Four fingers
            scrolling = True
            pyautogui.scroll(-5)  # Continuous scroll down using pyautogui

        else:
            scrolling = False  # Stop scrolling when the gesture is removed

        if not scrolling:
            if finger[1] == 1 and finger[2] == 0:
                x1, y1 = lmList[8][1:]
                x3 = np.interp(x1, (75, 640 - 75), (0, wScr))
                y3 = np.interp(y1, (75, 480 - 75), (0, hScr))

                cX = pX + (x3 - pX) / 7
                cY = pY + (y3 - pY) / 7

                autopy.mouse.move(wScr - cX, cY)
                pX, pY = cX, cY

            if finger[1] == 0 and finger[0] == 1:
                autopy.mouse.click()

            current_time = time.time()
            if finger[1] == 1 and finger[2] == 1 and (current_time - last_right_click_time) > CLICK_COOLDOWN:
                autopy.mouse.toggle(down=True, button=autopy.mouse.Button.RIGHT)
                autopy.mouse.toggle(down=False, button=autopy.mouse.Button.RIGHT)
                last_right_click_time = current_time

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
