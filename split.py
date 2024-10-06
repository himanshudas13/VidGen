import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

# Step 1: Load all the frames
def load_frames(directory):
    frames = []
    frame_files = sorted(os.listdir(directory))
    
    for file in frame_files:
        img = Image.open(os.path.join(directory, file)).convert('L')  # Convert to grayscale
        img_array = np.array(img)
        frames.append(img_array)
    
    frames = np.array(frames) / 255.0  # Normalize to [0, 1]
    return frames

# Step 2: Reshape frames into (rows, sequences, height, width, channels)
def reshape_frames(frames, sequence_length=20, num_sequences=80):
    total_frames = len(frames)
    usable_frames = min(total_frames, num_sequences * sequence_length)
    reshaped_frames = frames[:usable_frames].reshape((num_sequences, sequence_length, 64, 64, 1))
    return reshaped_frames

# Step 3: Create shifted frames for training and validation
def create_shifted_frames(data):
    x = data[:, 0:data.shape[1] - 1, :, :]
    y = data[:, 1:data.shape[1], :, :]
    return x, y

# Step 4: Load, reshape, and split data
def prepare_data(frame_dir='VidGen/frames_64', sequence_length=20, num_sequences=80):
    # Load frames
    frames = load_frames(frame_dir)
    # Reshape the frames
    reshaped_frames = reshape_frames(frames, sequence_length, num_sequences)
    
    # Split data into training and validation sets BEFORE creating shifted frames
    reshaped_train, reshaped_val = train_test_split(reshaped_frames, test_size=0.2, random_state=42)

    # Create shifted frames for both training and validation
    x_train, y_train = create_shifted_frames(reshaped_train)
    x_val, y_val = create_shifted_frames(reshaped_val)

    print(f"Training set shape (x_train, y_train): {x_train.shape}, {y_train.shape}")
    print(f"Validation set shape (x_val, y_val): {x_val.shape}, {y_val.shape}")
    
    return x_train, y_train, x_val, y_val

# Tester code
# frame_dir = 'VidGen/frames_64'  # Directory where frames are saved
# sequence_length = 20  # Length of each sequence
# num_sequences = 80    # Total number of sequences
# x_train, y_train, x_val, y_val = prepare_data(frame_dir, sequence_length, num_sequences)
