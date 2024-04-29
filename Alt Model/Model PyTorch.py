import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision.models.video import r3d_18, R3D_18_Weights


class SignLanguageDataset(Dataset):
    def __init__(self, video_paths, labels, frame_count=30, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.frame_count = frame_count
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def read_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret or len(frames) == self.frame_count:
                    break
                # Apply transformation and convert to tensor
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
        finally:
            cap.release()

        # Padding or slicing to ensure uniform frame count
        while len(frames) < self.frame_count:
            frames.append(torch.zeros_like(frames[0]))
        frames = frames[:self.frame_count]  # Ensure no excess frames if any

        # Stack the frames, should result in [frame_count, C, H, W]
        video_tensor = torch.stack(frames)
        # Permute to get [C, frame_count, H, W]
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        return video_tensor

    def __getitem__(self, idx):
        print(f"Loading item: {idx}")
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        video = self.read_video(video_path)
        print(f"Loaded item: {idx}")
        return video, label


def initialize_model(num_classes):
    weights = R3D_18_Weights.KINETICS400_V1  # Ensure using the appropriate weight enum
    model = r3d_18(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


# Load your data frame as previously
data_frame = pd.read_csv('top_50_glosses.csv', dtype={'video_id': str})
train_df, test_df = train_test_split(data_frame, test_size=0.2, random_state=42)
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df['gloss'])
test_labels = label_encoder.transform(test_df['gloss'])

train_dataset = SignLanguageDataset(
    train_df['video_id'].apply(lambda x: os.path.join('Top 50 Videos Processed', f'{x}.mp4')).tolist(),
    train_labels,
    transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
)
test_dataset = SignLanguageDataset(
    test_df['video_id'].apply(lambda x: os.path.join('Top 50 Videos Processed', f'{x}.mp4')).tolist(),
    test_labels,
    transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

for videos, labels in train_loader:
    print('Shape of video batch:', videos.shape)  # Should print: [batch_size, 3, 30, 256, 256]
    if videos.shape[1] != 3 or videos.shape[2] != 30:
        print("Error in video shapes:", videos.shape)
    break

# Test DataLoader
for i, (videos, labels) in enumerate(train_loader):
    print(f"Batch {i} loaded")
    if i == 1:  # Test with only two batches
        break


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for videos, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')


model = initialize_model(50)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer)


def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for videos, labels in test_loader:
            outputs = model(videos)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total}%')


evaluate_model(model, test_loader)

