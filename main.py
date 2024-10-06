from video import create_frames
from split import prepare_data
from plots import sample_train_plot
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


# generate frames from a sample video to create dataset
video_path = 'D:/CNN/VidGen/Puddles.mp4'
frame_dir = 'D:/CNN/VidGen/frames_64'        # Directory where frames are saved
create_frames(video_path,frame_dir)


# create  training and testing set 

# default input and output shape for the tarining and testing set
# sequence_length = 21  # 20 input frames, 1 output frame
# input_frames = 20  # Number of input frames
# img_size = (64, 64)  # The size of each image (64x64)

X_train, y_train, X_val, y_val=prepare_data(frame_dir)
# sample_train_plot(X_train)

model=load_model('D:/CNN/VidGen/model_name.h5')

example = X_val[np.random.choice(range(len(X_val)), size=1)[0]]

# Pick the first/last ten frames from the example.
frames = example[:10, ...]
original_frames = example[10:, ...]

# Predict a new set of 10 frames.
for _ in range(10):
    # Extract the model's prediction and post-process it.
    new_prediction = model.predict(np.expand_dims(frames, axis=0))
    new_prediction = np.squeeze(new_prediction, axis=0)
    predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)

    # Extend the set of prediction frames.
    frames = np.concatenate((frames, predicted_frame), axis=0)

# Construct a figure for the original and new frames.
fig, axes = plt.subplots(2, 9, figsize=(20, 4))

# Plot the original frames.
for idx, ax in enumerate(axes[0]):
    ax.imshow(np.squeeze(original_frames[idx]), cmap="gray")
    ax.set_title(f"Frame {idx + 11}")
    ax.axis("off")

# Plot the new frames.
new_frames = frames[10:, ...]
for idx, ax in enumerate(axes[1]):
    ax.imshow(np.squeeze(new_frames[idx]), cmap="gray")
    ax.set_title(f"Frame {idx + 11}")
    ax.axis("off")

# Display the figure.
plt.show()



