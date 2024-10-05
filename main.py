from video import create_frames
from split import prepare_data


# generate frames from a sample video to create dataset
video_path = 'VidGen\cartoon1.mp4'
frame_dir = 'VidGen/frames_64'         # Directory where frames are saved
create_frames(video_path,frame_dir)


# create  training and testing set 

# default input and output shape for the tarining and testing set
# sequence_length = 21  # 20 input frames, 1 output frame
# input_frames = 20  # Number of input frames
# img_size = (64, 64)  # The size of each image (64x64)

X_train, X_val, y_train, y_val=prepare_data(frame_dir)


