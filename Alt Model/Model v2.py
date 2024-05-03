import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def load_video(path, frame_count=30):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_capture = np.linspace(0, total_frames - 1, frame_count, dtype=int)

    frames = []
    current_frame_index = 0
    frame_id = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame_index in frames_to_capture:
                frame = cv2.resize(frame, (256, 256))
                frame = frame.astype('float32') / 255.0  # Normalize frames
                frames.append(frame)
                frame_id += 1
                if frame_id >= frame_count:
                    break
            current_frame_index += 1
    finally:
        cap.release()

    # Padding if video is shorter than frame_count
    while len(frames) < frame_count:
        frames.append(np.zeros((256, 256, 3), dtype=np.float32))

    return np.array(frames)

def generator(video_folder, csv_path, batch_size=5, frame_count=30):
    data = pd.read_csv(csv_path, dtype={'video_id': str})
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(data['gloss'])
    categorical_labels = to_categorical(encoded_labels)

    while True:
        for start in range(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            batch_videos = []
            batch_labels = []

            for i in range(start, end):
                path = os.path.join(video_folder, f"{data.iloc[i]['video_id']}.mp4")
                video = load_video(path, frame_count)
                batch_videos.append(video)
                batch_labels.append(categorical_labels[i])

            yield np.array(batch_videos, dtype='float32'), np.array(batch_labels)

def build_model():
    model = Sequential([
        TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(None, 256, 256, 3)),
        TimeDistributed(MaxPooling2D(2, 2)),
        TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
        TimeDistributed(MaxPooling2D(2, 2)),
        TimeDistributed(Flatten()),
        LSTM(50),
        Dense(50, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Parameters
video_folder = 'Top 50 Videos Processed'
csv_path = 'top_50_glosses.csv'
batch_size = 5
frame_count = 30

# Prepare generator
train_gen = generator(video_folder, csv_path, batch_size, frame_count)

# Build and train model
model = build_model()
model.summary()
model.fit(train_gen, steps_per_epoch=100, epochs=10, verbose=1)

