import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from pytorch_i3d import InceptionI3d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SignLanguageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.video_frame = pd.read_csv(csv_file, dtype={'video_id': str})
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = {label: idx for idx, label in enumerate(self.video_frame['gloss'].unique())}

    def __len__(self):
        return len(self.video_frame)

    def __getitem__(self, idx):
        video_id = str(self.video_frame.iloc[idx, 0])
        video_path = f'{self.root_dir}/{video_id}.mp4'
        label_str = self.video_frame.iloc[idx, 1]
        label_idx = self.label_map[label_str]
        frames = self.preprocess_video(video_path)

        if self.transform:
            frames = self.transform(frames)

        return {'video': frames, 'label': label_idx}

    def preprocess_video(self, clip_path, output_size=(224, 224), num_frames=64):
        cap = cv2.VideoCapture(clip_path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                resized_frame = cv2.resize(frame, output_size)
                frames.append(resized_frame)
        finally:
            cap.release()

        frames = np.array(frames)

        if frames.size == 0:
            frames = np.zeros((num_frames, *output_size, 3), dtype=np.float32)
        elif frames.shape[0] < num_frames:
            padding = np.zeros((num_frames - frames.shape[0], *output_size, 3), dtype=frames.dtype)
            frames = np.concatenate((frames, padding), axis=0)
        frames = frames[:num_frames]

        frames = frames.astype('float32') / 255.0
        frames = np.transpose(frames, (3, 0, 1, 2))
        return frames


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SignLanguageDataset(csv_file='../CSVs and JSONs/top_100_glosses.csv', root_dir='../Sign Language Videos')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    model = InceptionI3d(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Wrap the dataloader with tqdm for a progress bar
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for i, data in progress_bar:
            videos = data['video'].to(device)
            labels = data['label'].to(device)

            # Forward pass
            outputs = model(videos)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update running loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            progress_bar.set_postfix(loss=running_loss/(i+1), accuracy=100. * correct / total)

        # Epoch completion
        average_loss = running_loss / len(dataloader)
        accuracy = 100. * correct / total
        print(f'End of Epoch {epoch + 1}, Average Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')

    print("Training complete.")


if __name__ == '__main__':
    main()
