import cv2
import mediapipe as mp
import numpy as np
import os
import json
import pandas as pd
from multiprocessing import pool

VIDEO_FOLDER = 'Sign Language Videos'
CSV_FILE = 'CSVs and JSONs/video_ids_per_gloss.csv'
NUM_VIDEOS_TO_PROCESS = 10

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

all_videos_landmarks = []
all_labels = []

for video_file in np.random.choice(os.listdir(VIDEO_FOLDER), NUM_VIDEOS_TO_PROCESS, replace=False):
    video_path = os.path.join(VIDEO_FOLDER, video_file)
    video_id = int(os.path.splitext(video_file)[0])

    labels_df = pd.read_csv(CSV_FILE)

    video_label = labels_df.loc[labels_df['video_id'] == video_id, 'gloss'].iloc[0]

    cap = cv2.VideoCapture(video_path)
    video_landmarks = []

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

            RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(RGB_frame)

            landmarks_extracted = False

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmark_row = np.array(
                        [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark])
                    print("Shape of landmark_row:", landmark_row.shape)
                    video_landmarks.append(landmark_row)

                cv2.imshow("capture image", frame)

                print(f"Landmarks extracted for video {video_id}")

            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    all_videos_landmarks.append(video_landmarks)
    all_labels.append(video_label)

max_length = max(len(video_landmarks) for video_landmarks in all_videos_landmarks)

for i in range(len(all_videos_landmarks)):
    current_length = len(all_videos_landmarks[i])
    padding_needed = max_length - current_length
    padding = np.zeros((padding_needed, 21, 3))
    all_videos_landmarks[i] = np.concatenate([all_videos_landmarks[i], padding])

all_landmarks_array = np.array(all_videos_landmarks)
all_labels_array = np.array(all_labels)

print(all_landmarks_array.shape)

np.save('Arrays/10_landmarks.npy', all_landmarks_array)
np.save('Arrays/10_labels.npy', all_labels_array)

print("Processing completed for 10 Sign Language Videos.")
