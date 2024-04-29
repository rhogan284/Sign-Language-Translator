import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

def play_and_save_video(frames, output_path):
    """
    Play video frames and save the video to a file.

    Args:
        frames (numpy.ndarray): Video frames to display and save.
        output_path (str): Path where the video will be saved.
    """
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec (platform dependent)
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    for frame in frames:
        # Convert frame to BGR format before writing
        bgr_frame = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        out.write(bgr_frame)
        cv2.imshow('Video', cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
        if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    # Release everything when the job is finished
    out.release()
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
    frames = frames.astype('float32')
    return frames

# Example usage
clip_path = '../Top 100 Videos/00623.mp4'
output_path = 'Top 100 Videos Processed/output_video.mp4'
processed_frames = preprocess_video(clip_path)
play_and_save_video(processed_frames, output_path)
