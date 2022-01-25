import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 10)
mpDraw = mp.solutions.drawing_utils

pTime = 0 # previous time
cTime = 0 # current time

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)
    print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                # if id == 4:
                    # draw circle
                cv2.circle(img, (cx, cy),
                           25, # radius
                           (255, 0, 255), #circle
                           cv2.FILLED #fill mode
                )

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(
        img,  # from image source
        str(int(fps)), # add text
        (10, 70), # move x, y position
        cv2.FONT_HERSHEY_PLAIN,  # font style
        3, # font size
        (255, 0, 255), # font color
        3 # font thickness
    )

    cv2.imshow("Image", img)
    cv2.waitKey(1)