import cv2

video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    print("Video is open")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("No more frames or cannot fetch frames.")
        break
    cv2.imshow('Video', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    count += 1

print(f"Frames read: {count}")
cap.release()
cv2.destroyAllWindows()
