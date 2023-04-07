import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

# hand detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# drawing keypoints
mpDraw = mp.solutions.drawing_utils

if not cap.isOpened():
    print("Cannot Open Camera")
    exit()

while True:
    success, img = cap.read()
    results = hands.process(img)
    if hand_landmarks:=results.multi_hand_landmarks:
        for handLms in hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(img, (cx, cy), 3, (255,0,255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    cv2.imshow("Image", img)
    cv2.waitKey(1)