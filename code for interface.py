from tkinter import *
from tkmacosx import Button
import cv2

root = Tk()
root.title("Talk2TheHand Interface")

my_label = Label(root, text="Welcome to Talk2TheHand")
label_instr = Label(root, text="Click the 'Begin' button to start the recording, and 'End' button to stop recording")


def end_camera():
    cv2.waitKey == button_end


def run_camera():
    camera = cv2.VideoCapture(0)

    # if statement that informs us whether the camera could open successfully or not
    if (camera.isOpened()):
        print("The camera has opened successfully")
    else:
        print("Could not open camera")

    # Need to determine what the frame width, height and rate is.
    # This block of code defines the flags we need to reteiev
    frameWidith = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameRate = int(camera.get(cv2.CAP_PROP_FPS))

    # Specify the video properties and desired video codec
    # Also need to specify the FourCC 4-byte-code

    # HEVC are used for .avi files
    # MJPG are used for .mp4 files

    fourccCode = cv2.VideoWriter_fourcc(*'HEVC')

    # Specify video file name
    videoFileName = 'recordedVideo.mp4'

    # Define the recorded video dimensions - done by pulling from our previouslu defined objects
    videoDimensions = (frameWidith, frameHeight)

    # Create a VideoWriter objects. Contains 3 inputs
    recordedVideo = cv2.VideoWriter(videoFileName,
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

        # the 'q' button is set as the quitting button you may use any desired button of your choice
        # need to tyr and change this q to the button_end
        if cv2.waitKey() == ord(end_camera):
            break

    recordedVideo.release()
    camera.release()
    cv2.destroyAllWindows()


button_start = Button(root, text="Begin", bg="orange", fg="white", command=run_camera)
button_end = Button(root, text="End", bg="orange", fg="white")
my_label.pack()
label_instr.pack()
button_start.pack()
button_end.pack()

root.mainloop()
