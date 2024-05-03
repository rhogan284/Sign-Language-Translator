from tkinter import *
from tkmacosx import Button
import cv2
from PIL import Image, ImageTk

root = Tk()
root.title("Talk2TheHand Interface")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}")

bg_color = "#333333"
btn_color = "#FF5733"
text_color = "#FFFFFF"
font_style = ("Arial", 12, "bold")

root.configure(bg=bg_color)

my_label = Label(root, text="Welcome to Talk2TheHand", bg=bg_color, fg=text_color, font=("Arial", 16, "bold"))
label_instr = Label(root, text="Click the 'Begin' button to start the recording, and 'End' button to stop recording.", bg=bg_color, fg=text_color, font=font_style)

camera = None
recordedVideo = None
running = False
video_frame_width = 1280
video_frame_height = 720
video_frame_label = Label(root, bg=bg_color)

def clear_window():
    for widget in root.winfo_children():
        widget.destroy()

def end_camera():
    global running
    running = False
    release_resources()
    show_processing_screen()

def release_resources():
    global recordedVideo, camera
    if recordedVideo is not None:
        recordedVideo.release()
        recordedVideo = None
    if camera is not None:
        camera.release()
        camera = None
    cv2.destroyAllWindows()

def show_processing_screen():
    clear_window()
    processing_label = Label(root, text="Processing video...", bg=bg_color, fg=text_color, font=("Arial", 16, "bold"))
    processing_label.pack(expand=True)

def run_camera():
    global camera, recordedVideo, running
    if not running:
        camera = cv2.VideoCapture(0)
        running = True
        if not camera.isOpened():
            print("Could not open camera")
            running = False
            return
        frameWidth = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frameRate = int(camera.get(cv2.CAP_PROP_FPS))
        fourccCode = cv2.VideoWriter_fourcc(*'mp4v')
        videoFileName = 'test.mp4'
        videoDimensions = (frameWidth, frameHeight)
        recordedVideo = cv2.VideoWriter(videoFileName, fourccCode, frameRate, videoDimensions)
    read_frame()

def read_frame():
    if running:
        success, frame = camera.read()
        if success:
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (video_frame_width, video_frame_height))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image_tk = ImageTk.PhotoImage(image=image)
            video_frame_label.imgtk = image_tk
            video_frame_label.config(image=image_tk)
            recordedVideo.write(frame)
        root.after(10, read_frame)
    else:
        release_resources()


button_frame = Frame(root, bg=bg_color)
button_start = Button(button_frame, text="Begin", bg=btn_color, fg=text_color, font=font_style, borderless=1, command=run_camera)
button_end = Button(button_frame, text="End", bg=btn_color, fg=text_color, font=font_style, borderless=1, command=end_camera)

my_label.pack(pady=(20, 0))
label_instr.pack(pady=(5, 0))
button_frame.pack(pady=(5, 20))
button_start.pack(side='left', padx=10)
button_end.pack(side='right', padx=10)
video_frame_label.pack(pady=(10, 20), fill='both', expand=True)

root.mainloop()

