import torch
import torch.nn.functional as F
from pytorch_i3d import InceptionI3d
import cv2
import numpy as np
import os

def preprocess_video(video_path, target_height=400, target_width=720):
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames = []
    success, img = vidcap.read()
    while success:
        # Resize and convert to RGB format
        img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
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


video_path = "../Sign Language Videos/57278.mp4"
frames = preprocess_video(video_path)

model_path = 'archived/asl100/FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt'
num_classes = 100
i3d = InceptionI3d(num_classes)
i3d.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
i3d.eval()

print(f"Frame shape before model input: {frames.shape}")
print(f"Data type and range: {frames.dtype}, {frames.min()} - {frames.max()}")

if frames is not None:
    with torch.no_grad():
        outputs = i3d(frames)
        print(f"Output shape before torch.max: {outputs.shape}")
        predictions = torch.max(outputs, dim=2)[0]
else:
    print(f"Could not load video frames from: {video_path}")

class_labels = []
with open("preprocess/wlasl_class_list_100.txt", 'r') as f:
    for line in f:
        _, label = line.strip().split('\t')
        class_labels.append(label)

reduced_outputs = outputs.mean(dim=(2, 3, 4))

print(f"Outputs shape before softmax: {reduced_outputs.shape}")

probabilities = F.softmax(reduced_outputs, dim=1)
topk_values, topk_indices = torch.topk(probabilities, 10, dim=1, largest=True, sorted=True)
top_predictions = [class_labels[idx] for idx in topk_indices[0].tolist()]

print("Top 10 Predictions and their Probabilities:")
for i, (prob, label) in enumerate(zip(topk_values[0], top_predictions)):
    print(f"{i+1}: {label} with probability {prob.item():.4f}")
