import cv2
import mediapipe as mp
import numpy as np
import os
import json
import pandas as pd
from multiprocessing import Pool

VIDEO_FOLDER = 'Sign Language Videos'
CSV_FILE = 'CSVs and JSONs/Video_Ids_Final.csv'
NUM_GLOSSES_TO_SELECT = 100
NUM_VIDEOS_PER_GLOSS = 10

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

labels_df = pd.read_csv(CSV_FILE)

unique_glosses = labels_df['gloss'].unique()

selected_videos = []
selected_glosses = []

while len(selected_glosses) < NUM_GLOSSES_TO_SELECT:
    gloss = np.random.choice(unique_glosses, 1, replace=False)[0]
    if gloss not in selected_glosses:
        videos_for_gloss = labels_df[labels_df['gloss'] == gloss]['video_id'].tolist()
        if len(videos_for_gloss) >= NUM_VIDEOS_PER_GLOSS:
            selected_videos.extend(np.random.choice(videos_for_gloss, NUM_VIDEOS_PER_GLOSS, replace=False))
            selected_glosses.append(gloss)

print(f"Total selected videos: {len(selected_videos)}")

def process_video(video_id):
    video_file = str(video_id).zfill(5) + '.mp4'
    video_path = os.path.join(VIDEO_FOLDER, video_file)
    if not os.path.exists(video_path):
        print(f"File does not exist: {video_path}")

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
                print(f"Video {video_id} Ended")
                break

            RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(RGB_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmark_row = np.array(
                        [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark])
                    video_landmarks.append(landmark_row)

            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    return video_landmarks, video_label

if __name__ == '__main__':

    with Pool(processes=min(len(selected_videos), os.cpu_count())) as pool:
        results = pool.map(process_video, selected_videos)

    all_videos_landmarks = [result[0] for result in results]
    all_labels = [result[1] for result in results]

    max_length = max(len(video_landmarks) for video_landmarks in all_videos_landmarks)
    for i in range(len(all_videos_landmarks)):
        current_length = len(all_videos_landmarks[i])
        padding_needed = max_length - current_length
        padding = np.zeros((padding_needed, 21, 3))
        all_videos_landmarks[i] = np.concatenate([all_videos_landmarks[i], padding])

    all_landmarks_array = np.array(all_videos_landmarks)
    all_labels_array = np.array(all_labels)

    np.save('Arrays/1000_landmarks.npy', all_landmarks_array)
    np.save('Arrays/1000_labels.npy', all_labels_array)

    print(f"Processing completed for {len(selected_videos)} Sign Language Videos.")
