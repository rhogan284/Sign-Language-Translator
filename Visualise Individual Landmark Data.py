import cv2
import numpy as np

landmarks_array = np.load('Arrays/hand_landmarks.npy')

width = 640
height = 480
fps = 30

out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

for landmarks_frame in landmarks_array:
    print("Shape of landmarks_frame:", landmarks_frame.shape)
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    for landmark in landmarks_frame:
        x, y, _ = landmark
        x = int(x * width)
        y = int(y * height)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    out.write(frame)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
