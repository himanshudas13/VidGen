import cv2
from PIL import Image
import os


def create_frames(video_path,output_dir = 'frames_64'):
   
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    while cap.isOpened():
       ret, frame = cap.read()
      
       if not ret:
           break
      # Convert frame to RGB (OpenCV reads in BGR by default)
      
       frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      # Convert to PIL Image for resizing
       img = Image.fromarray(frame_rgb)
    
      # Resize to 64x64
       img_resized = img.resize((64, 64), Image.Resampling.LANCZOS)
    
      # Save the frame as an image
       frame_filename = os.path.join(output_dir, f'frame_{frame_num:04d}.png')
       img_resized.save(frame_filename)
    
       frame_num += 1
    cap.release()
    print(f"Extracted and resized {frame_num} frames.")
    
# tester Code
# output_dir = 'VidGen/frames_64'
# video_path = 'VidGen\cartoon1.mp4'
# create_frames(video_path,output_dir)
    
    
