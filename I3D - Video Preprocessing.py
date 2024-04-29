import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

def play_video(frames):
    for frame in frames:
        cv2.imshow('Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to exit
            break
    cv2.destroyAllWindows()

def preprocess_video(clip_path, output_size=(224, 224), num_frames=64):
    """
    Extract frames from video, resize, and sample a fixed number of frames.

    Args:
        clip_path (str): Path to the video file.
        output_size (tuple): Target size (width, height) for each frame.
        num_frames (int): Number of frames to sample uniformly from the video.

    Returns:
        numpy.ndarray: Array of shape (num_frames, height, width, 3) containing the preprocessed video frames.
    """
    cap = cv2.VideoCapture(clip_path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, output_size)
            frames.append(resized_frame)
    finally:
        cap.release()

    frames = np.array(frames)
    # Uniformly sample frames
    if len(frames) > num_frames:
        indices = np.linspace(0, len(frames), num_frames, endpoint=False, dtype=int)
        frames = frames[indices]

    # Normalize pixel values to [0, 1]
    frames = frames.astype('float32') / 255.0
    return frames


# Example usage
clip_path = 'archive/videos/0033.mp4'
processed_frames = preprocess_video(clip_path)
play_video(processed_frames)
