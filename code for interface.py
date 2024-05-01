from tkinter import *
from tkmacosx import Button
import cv2

root = Tk()
root.title("Talk2TheHand Interface")

my_label = Label(root, text="Welcome to Talk2TheHand")
label_instr = Label(root, text="Click the 'Begin' button to start the recording, and 'End' button to stop recording")

camera = None
recordedVideo = None

def end_camera():
    global recordedVideo, camera
    if recordedVideo is not None:
        recordedVideo.release()
    if camera is not None:
        camera.release()
    cv2.destroyAllWindows()

def run_camera():
    global camera, recordedVideo
    camera = cv2.VideoCapture(0)

    if camera.isOpened():
        print("The camera has opened successfully")
    else:
        print("Could not open camera")

    frameWidth = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameRate = int(camera.get(cv2.CAP_PROP_FPS))

    fourccCode = cv2.VideoWriter_fourcc(*'mp4v')
    videoFileName = 'recordedVideo.mp4'
    videoDimensions = (frameWidth, frameHeight)

    recordedVideo = cv2.VideoWriter(videoFileName, fourccCode, frameRate, videoDimensions)

    while True:
        success, frame = camera.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        cv2.imshow('Frame', frame)
        recordedVideo.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop when 'q' is pressed
            break


button_start = Button(root, text="Begin", bg="orange", fg="white", command=run_camera)
button_end = Button(root, text="End", bg="orange", fg="white", command=end_camera)

my_label.pack()
label_instr.pack()
button_start.pack()
button_end.pack()

root.mainloop()
