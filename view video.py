import cv2

# Path to the video file
video_path = 'recordedVideo.mp4'

# Create a VideoCapture object and open the input file
cap = cv2.VideoCapture(video_path)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
