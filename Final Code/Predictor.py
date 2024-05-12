import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pytorch_i3d import InceptionI3d

def preprocess_video(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success, img = vidcap.read()
    frames = []
    while success:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
        success, img = vidcap.read()

    if not frames:
        raise ValueError(f"Could not read any frames from video file: {video_path}")

    # Stack and normalize frames
    frames = np.stack(frames, axis=0)
    frames = frames.astype(np.float32) / 255.0  # Normalize to [0, 1]
    frames = np.transpose(frames, (3, 0, 1, 2))  # Corrected transposition
    frames = frames[np.newaxis, ...]  # Add a batch dimension
    frames = torch.from_numpy(frames).float()

    return frames

def process_video(video_path):
    frames = preprocess_video(video_path)
    model_path = 'archived/asl100/FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt'
    num_classes = 100
    i3d = InceptionI3d(num_classes)
    i3d.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    i3d.eval()

    if frames is not None:
        with torch.no_grad():
            outputs = i3d(frames)
        reduced_outputs = outputs.mean(dim=(2, 3, 4))
        probabilities = F.softmax(reduced_outputs, dim=1)
        topk_values, topk_indices = torch.topk(probabilities, 10, dim=1, largest=True, sorted=True)
        class_labels = []
        with open("wlasl_class_list_100.txt", 'r') as f:
            for line in f:
                _, label = line.strip().split('\t')
                class_labels.append(label)
        top_predictions = [(class_labels[idx], prob.item()) for idx, prob in zip(topk_indices[0], topk_values[0])]
        return top_predictions
    else:
        return ["Could not load video frames from: " + video_path]

# Example use case (This line is for testing and should be commented or removed in actual deployment)
# predictions = process_video("path_to_video_file.mp4")
# print(predictions)
