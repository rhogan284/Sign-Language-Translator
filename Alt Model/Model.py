import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, TimeDistributed, LSTM, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers.legacy import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import cv2
import os

# Constants
FRAME_WIDTH = 256
FRAME_HEIGHT = 256
CHANNELS = 3  # RGB channels
MOBILENET_WEIGHTS_PATH = 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'

labels_path = 'top_50_glosses.csv'
labels_df = pd.read_csv(labels_path, dtype={'video_id': str})

# Encode the labels
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels_df['gloss'])
labels_one_hot = to_categorical(integer_encoded)

# Path to the video folder
video_folder = 'Top 50 videos processed'  # Update this path accordingly

# Function to load and preprocess video frames
# Function to load and preprocess video frames
def load_video(video_id, frame_count=30):
    video_path = os.path.join(video_folder, f'{video_id}.mp4')
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip = np.linspace(0, total_frames - 1, frame_count, dtype=int) if total_frames > frame_count else np.arange(total_frames)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in skip:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                frames.append(frame)
            frame_idx += 1
    finally:
        cap.release()
    return np.array(frames)


# Load and preprocess all videos as a list
video_data = [load_video(vid) for vid in labels_df['video_id']]

X_train, X_test, y_train, y_test = train_test_split(video_data, labels_one_hot, test_size=0.2, random_state=42)

print("Data Loaded")

def video_batch_generator(video_data, labels, batch_size=32):
    while True:
        for start in range(0, len(video_data), batch_size):
            end = min(start + batch_size, len(video_data))
            batch_videos = [v for v in video_data[start:end]]
            max_frames = max(v.shape[0] for v in batch_videos)  # Find the max number of frames in the batch
            padded_videos = [np.pad(v, ((0, max_frames - v.shape[0]), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0) for v in batch_videos]
            batch_labels = np.array(labels[start:end])
            yield np.array(padded_videos), batch_labels



# Load the pre-trained MobileNetV2 model without the top layer
base_model = MobileNetV2(weights=None, include_top=False,
                         input_shape=(FRAME_HEIGHT, FRAME_WIDTH, CHANNELS))
base_model.load_weights(MOBILENET_WEIGHTS_PATH)

# Freeze the layers of the base model to prevent them from being trained
for layer in base_model.layers:
    layer.trainable = False

# Model architecture
model = Sequential([
    InputLayer(input_shape=(None, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS), dtype=tf.float32, ragged=True),
    TimeDistributed(base_model),
    TimeDistributed(tf.keras.layers.GlobalAveragePooling2D()),

    LSTM(256, return_sequences=False),
    tf.keras.layers.Dropout(0.5),
    Dense(50, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Model Compiled")
# Configure the model's fit function to use the generator
train_generator = video_batch_generator(X_train, y_train)
validation_generator = video_batch_generator(X_test, y_test)
print("Model generator configured")

# Determine steps per epoch based on the batch size
steps_per_epoch = len(X_train) // 32
validation_steps = len(X_test) // 32

test_gen = video_batch_generator(X_train, y_train, batch_size=1)
samples, labels = next(test_gen)
print("Batch shape:", samples.shape)  # Should now correctly show ragged dimensions
print("Labels shape:", labels.shape)

# Try running one training step
outputs = model(samples, training=True)
print("Model output shape:", outputs.shape)

# Fit the model
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps,
                    epochs=10, verbose=1)  # You can adjust the number of epochs
