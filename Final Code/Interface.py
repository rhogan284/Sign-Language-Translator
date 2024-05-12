from tkinter import *
from tkmacosx import Button
import cv2
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
from I3D.pytorch_i3d import InceptionI3d
from Predictor import preprocess_video
import threading

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

camera = None
running = False
recordedVideo = None

video_frame_width = 1280
video_frame_height = 720
video_frame_label = Label(root, bg=bg_color)

model_path = '../Final Code/archived/asl100/FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt'
num_classes = 100
i3d = InceptionI3d(num_classes)
i3d.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
i3d.eval()

class_labels = []
with open("capitalized_wlasl_class_list_100.txt", 'r') as f:
    for line in f:
        _, label = line.strip().split('\t')
        class_labels.append(label)

def clear_window():
    for widget in root.winfo_children():
        widget.destroy()

def reset_interface():
    clear_window()
    global camera, recordedVideo, running
    release_resources()
    initialize_gui()

def exit_interface():
    global root
    root.quit()


def initialize_gui():
    global my_label, label_instr, button_frame, button_start, button_end, button_reset, button_exit, video_frame_label, top_button_frame
    top_frame = Frame(root, bg=bg_color)
    top_frame.pack(fill='x', pady=(10, 0))

    left_space = Frame(top_frame, width=20, bg=bg_color)
    left_space.pack(side='left', fill='y')

    button_reset = Button(top_frame, text="Reset", bg=btn_color, fg=text_color, font=font_style, borderless=1,
                          command=reset_interface)
    button_reset.pack(side='left', padx=10)

    my_label = Label(top_frame, text="Welcome to Talk2TheHand", bg=bg_color, fg=text_color, font=("Arial", 16, "bold"))
    my_label.pack(side='left', expand=True)

    button_exit = Button(top_frame, text="Exit", bg=btn_color, fg=text_color, font=font_style, borderless=1,
                         command=exit_interface)
    button_exit.pack(side='right', padx=10)

    right_space = Frame(top_frame, width=20, bg=bg_color)
    right_space.pack(side='right', fill='y')

    label_instr = Label(root,
                        text="Click the 'Begin' button to start the recording, and 'End' button to stop recording.",
                        bg=bg_color, fg=text_color, font=font_style)
    label_instr.pack(pady=(10, 0))

    button_frame = Frame(root, bg=bg_color)
    button_frame.pack(pady=(10, 20))

    button_start = Button(button_frame, text="Begin", bg=btn_color, fg=text_color, font=font_style, borderless=1,
                          command=run_camera)
    button_end = Button(button_frame, text="End", bg=btn_color, fg=text_color, font=font_style, borderless=1,
                        command=end_camera)
    button_start.pack(side='left', padx=10)
    button_end.pack(side='right', padx=10)

    video_frame_label = Label(root, bg=bg_color)
    video_frame_label.pack(pady=(10, 20), fill='both', expand=True)


def process_video():
    frames = preprocess_video("recordedVideo.mp4")
    predictions, probabilities = run_prediction(frames)
    display_prediction_window(predictions, probabilities)

def end_camera():
    global running, recordedVideo
    running = False
    if recordedVideo:
        recordedVideo.release()
        recordedVideo = None
    release_resources()
    show_processing_screen()
    threading.Thread(target=process_video).start()

def release_resources():
    global recordedVideo, camera
    if recordedVideo is not None:
        recordedVideo.release()
        recordedVideo = None
    if camera is not None:
        camera.release()
        camera = None
    cv2.destroyAllWindows()

def run_prediction(frames):
    with torch.no_grad():
        outputs = i3d(frames)
        probabilities = F.softmax(outputs, dim=1)
        topk_values, topk_indices = torch.topk(probabilities, 10, dim=1, largest=True, sorted=True)
        top_predictions = [class_labels[idx] for idx in topk_indices[0].tolist()]
    return top_predictions, topk_values[0]

def show_processing_screen():
    global button_reset, button_exit
    clear_window()

    top_frame = Frame(root, bg=bg_color)
    top_frame.pack(fill='x', pady=(10, 0))

    left_space = Frame(top_frame, width=20, bg=bg_color)
    left_space.pack(side='left', fill='y')

    button_reset = Button(top_frame, text="Reset", bg=btn_color, fg=text_color, font=font_style, borderless=1,
                          command=reset_interface)
    button_reset.pack(side='left', padx=10)

    button_exit = Button(top_frame, text="Exit", bg=btn_color, fg=text_color, font=font_style, borderless=1,
                         command=exit_interface)
    button_exit.pack(side='right', padx=10)

    right_space = Frame(top_frame, width=20, bg=bg_color)
    right_space.pack(side='right', fill='y')

    processing_label = Label(root, text="Processing video...", bg=bg_color, fg=text_color, font=("Arial", 16, "bold"))
    processing_label.pack(expand=True)

def display_prediction_window(predictions, probabilities):
    global button_reset, button_exit
    clear_window()

    top_frame = Frame(root, bg=bg_color)
    top_frame.pack(fill='x', pady=(10, 0))

    left_space = Frame(top_frame, width=20, bg=bg_color)
    left_space.pack(side='left', fill='y')

    button_reset = Button(top_frame, text="Reset", bg=btn_color, fg=text_color, font=font_style, borderless=1,
                          command=reset_interface)
    button_reset.pack(side='left', padx=10)

    button_exit = Button(top_frame, text="Exit", bg=btn_color, fg=text_color, font=font_style, borderless=1,
                         command=exit_interface)
    button_exit.pack(side='right', padx=10)

    right_space = Frame(top_frame, width=20, bg=bg_color)
    right_space.pack(side='right', fill='y')

    prediction_label = Label(root, text="Top 10 Predictions and their Probabilities:", bg=bg_color, fg=text_color, font=("Arial", 14, "bold"))
    prediction_label.pack(pady=10)

    for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
        label = Label(root, text=f"{i+1}: {pred} with probability {prob.item():.4f}", bg=bg_color, fg=text_color, font=font_style)
        label.pack()

def run_camera():
    global camera, recordedVideo, running
    if not running:
        camera = cv2.VideoCapture(2)
        running = True
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
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image_tk = ImageTk.PhotoImage(image=image)
            video_frame_label.imgtk = image_tk
            video_frame_label.config(image=image_tk)
            recordedVideo.write(frame)
        root.after(10, read_frame)
    else:
        release_resources()


initialize_gui()
root.mainloop()