from tkinter import *
from tkmacosx import Button
import cv2

root = Tk()
root.title("Talk2TheHand Interface")

my_label = Label(root, text="Welcome to Talk2TheHand")
label_instr = Label(root, text="Click the 'Begin' button to start the recording, and 'End' button to stop recording.")

camera = None
recordedVideo = None
running = False  # Control flag for the camera

def end_camera():
    global running
    running = False  # Stop recording

def release_resources():
    global recordedVideo, camera
    if recordedVideo is not None:
        recordedVideo.release()
        recordedVideo = None
    if camera is not None:
        camera.release()
        camera = None
    cv2.destroyAllWindows()

def run_camera():
    global camera, recordedVideo, running
    if not running:
        camera = cv2.VideoCapture(0)  # Initialize camera
        running = True  # Set running to True to start the loop

        if not camera.isOpened():
            print("Could not open camera")
            running = False
            return

        frameWidth = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frameRate = int(camera.get(cv2.CAP_PROP_FPS))

        fourccCode = cv2.VideoWriter_fourcc(*'mp4v')
        videoFileName = 'recordedVideo.mp4'
        videoDimensions = (frameWidth, frameHeight)

        recordedVideo = cv2.VideoWriter(videoFileName, fourccCode, frameRate, videoDimensions)

    read_frame()

def read_frame():
    if running:
        success, frame = camera.read()
        if success:
            frame = cv2.flip(frame, 1)
            cv2.imshow('Frame', frame)
            recordedVideo.write(frame)
        root.after(10, read_frame)  # Fetch next frame after 10 ms
    else:
        release_resources()

button_start = Button(root, text="Begin", bg="orange", fg="white", command=run_camera)
button_end = Button(root, text="End", bg="orange", fg="white", command=end_camera)

my_label.pack()
label_instr.pack()
button_start.pack()
button_end.pack()

root.mainloop()
