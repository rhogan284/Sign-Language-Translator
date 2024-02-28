import cv2
import numpy as np

# Load the saved hand landmarks
landmarks_array = np.load('hand_landmarks.npy')

print(landmarks_array.shape)

# Define the blank video properties (width, height, and frames per second)
width = 640
height = 480
fps = 30

# Create a VideoWriter object to save the output video
out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

# Iterate through each frame
for landmarks_frame in landmarks_array:
    # Create a blank frame
    print("Shape of landmarks_frame:", landmarks_frame.shape)
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw landmarks on the blank frame
    for landmark in landmarks_frame:
        x, y, _ = landmark
        x = int(x * width)
        y = int(y * height)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle for each landmark

    # Write the frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoWriter and destroy all OpenCV windows
out.release()
cv2.destroyAllWindows()
