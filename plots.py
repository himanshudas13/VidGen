import numpy as np
import matplotlib.pyplot as plt
# from main import X_train

import matplotlib.pyplot as plt
import numpy as np

# Assuming x_train is defined and has the correct shape
# Pick a random index

def sample_train_plot(X_train):
    random_index = np.random.randint(0, X_train.shape[0])

    # Get the frames from the random example
    random_example_frames = X_train[random_index]

    # Number of frames in the sequence
    num_frames = random_example_frames.shape[0]

    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_frames, figsize=(15, 5))

    # Plot each frame in the random example
    for i in range(num_frames):
      axes[i].imshow(random_example_frames[i].squeeze(), cmap='gray')  # Squeeze to remove the channel dimension
      axes[i].axis('off')  # Hide the axis

    plt.suptitle(f'Frames from Random Training Example {random_index}')
    plt.show() 
    
    