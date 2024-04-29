import cv2
import numpy as np

landmarks_array = np.load('../Arrays/1200_landmarks.npy')
labels_array = np.load('../Arrays/1200_labels.npy')

print(landmarks_array.shape)

width = 640
height = 480
target_fps = 30

dot_color = (0, 0, 255)
dot_size = 3
line_color = (0, 255, 0)
line_thickness = 2

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]

for video_index, landmarks_video in enumerate(landmarks_array):
    label = labels_array[video_index]

    for landmarks_frame in landmarks_video:
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        for landmark in landmarks_frame:
            x, y, _ = landmark
            x = int(x * width)
            y = int(y * height)

            cv2.circle(frame, (x, y), dot_size, dot_color, -1)

            for connection in connections:
                start_idx, end_idx = connection
                start_point = (int(landmarks_frame[start_idx][0] * width), int(landmarks_frame[start_idx][1] * height))
                end_point = (int(landmarks_frame[end_idx][0] * width), int(landmarks_frame[end_idx][1] * height))
                cv2.line(frame, start_point, end_point, line_color, line_thickness)

        display_delay = int(500 / target_fps)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(display_delay) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
