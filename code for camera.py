import cv2

camera = cv2.VideoCapture(0)

#if statement that informs us whether the camera could open successfully or not
if (camera.isOpened()):
    print("The camera has opened successfully")
else:
    print("Could not open camera")

#Need to determine what the frame width, height and rate is.
# This block of code defines the flags we need to reteiev
frameWidith=int(camera.get(cv2.CAP_PROP_FRAME_WIDTH ))
frameHeight=int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameRate=int(camera.get(cv2.CAP_PROP_FPS))


#Specify the video properties and desired video codec
#Also need to specify the FourCC 4-byte-code

# HEVC are used for .avi files
# MJPG are used for .mp4 files

fourccCode=cv2.VideoWriter_fourcc(*'HEVC')

#Specify video file name
videoFileName='recordedVideo.mp4'

# Define the recorded video dimensions - done by pulling from our previouslu defined objects
videoDimensions=(frameWidith, frameHeight)

# Create a VideoWriter objects. Contains 3 inputs
recordedVideo=cv2.VideoWriter(videoFileName,
                              fourccCode,
                              frameRate,
                              videoDimensions)


while (True):
        # Capture the video frame
        # by frame
    success, frame = camera.read()

        # Display the resulting frame
    cv2.imshow('Frame', frame)
    recordedVideo.write(frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

recordedVideo.release()
camera.release()
cv2.destroyAllWindows()

