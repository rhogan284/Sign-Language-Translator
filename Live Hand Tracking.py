import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2 ,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Ignoring empty frame")
            continue

        frame = cv2.flip(frame, 1)
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(RGB_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                print(hand_landmarks)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("capture image", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
