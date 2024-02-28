import cv2
import mediapipe as mp
import numpy as np

# Load hand landmarks
landmarks_array = np.load('hand_landmarks.npy')
print(landmarks_array.shape)

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Video setup
width = 1280  # Example resolution
height = 720
fps = 30  # Adjust if needed to match your original video
output_video = cv2.VideoWriter('landmarks_visualization.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    for frame_landmarks_list in landmarks_array:
        for hand_landmarks in frame_landmarks_list:  # Extract the hand landmarks directly
            blank_frame = np.zeros((height, width, 3), np.uint8)

            if hand_landmarks is None:
                # No hands detected
                pass
            elif isinstance(hand_landmarks, np.ndarray):  # Check if hand_landmarks is a numpy array
                print(hand_landmarks.shape)  # Debug print to check the shape
                print(hand_landmarks)  # Debug print to check the content

                # Construct a list of HandLandmark objects
                landmark_list = [mp_hands.HandLandmark(x=landmark[0], y=landmark[1], z=landmark[2]) for landmark in
                                 hand_landmarks]
                mp_drawing.draw_landmarks(blank_frame, landmark_list, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Landmark Visualization", blank_frame)
            output_video.write(blank_frame)

            if cv2.waitKey(1) == ord('q'):
                break

output_video.release()
cv2.destroyAllWindows()
