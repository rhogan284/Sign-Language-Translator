import numpy as np
from sklearn.model_selection import train_test_split

# Load the numpy arrays
labels = np.load('Arrays/1000_labels.npy')
landmarks = np.load('Arrays/1000_landmarks.npy')

X_train, X_test, y_train, y_test = [], [], [], []

# Unique labels
unique_labels = np.unique(labels)

for label in unique_labels:
    # Indices where current label occurs
    indices = np.where(labels == label)[0]

    # Split these indices into test and train, with test_size set to approximate ratio for 5:1 split
    # This is a simple way to ensure that for each label, we attempt to have 5 videos for testing and 1 for training
    train_idx, test_idx = train_test_split(indices, test_size=2 / 10, random_state=42)

    # Append the corresponding data to our lists
    X_train.extend(landmarks[train_idx])
    X_test.extend(landmarks[test_idx])
    y_train.extend(labels[train_idx])
    y_test.extend(labels[test_idx])

# Convert lists back to numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

unqiue_values, counts = np.unique(y_train, return_counts=True)

for value, count in zip(unqiue_values, counts):
    print(f'{value}: {count}')

np.save('Arrays/1000_landmarks_train.npy', X_train)
np.save('Arrays/1000_landmarks_test.npy', X_test)
np.save('Arrays/1000_labels_train.npy', y_train)
np.save('Arrays/1000_labels_test.npy', y_test)

print("All Arrays Saved Successfully")