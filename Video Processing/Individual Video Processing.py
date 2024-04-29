import cv2
import mediapipe as mp
import numpy as np

video_path = '../Sign Language Videos/18290.mp4'
cap = cv2.VideoCapture(video_path)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

all_landmarks = []

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Video Ended")
            break

        frame = cv2.flip(frame, 1)

        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(RGB_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmark_row = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark])

                print("Shape of landmark_row:", landmark_row.shape)
                all_landmarks.append(landmark_row)

            cv2.imshow("capture image", frame)

            if cv2.waitKey(1) == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()

landmarks_array = np.stack(list(all_landmarks))
print(landmarks_array.shape)
np.save('../Arrays/hand_landmarks.npy', landmarks_array)
