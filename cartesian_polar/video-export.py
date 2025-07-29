import cv2
import os

# Set the path to your frames
frames_folder = './frames'
output_video = 'output_video.mp4'
fps = 3  # Change as needed

# Get a sorted list of all frame filenames
frame_files = [f for f in os.listdir(frames_folder) if f.endswith(('.png', '.jpg'))]

# Sort based on numbers in filename (assumes format like 'frame_001.png')
frame_files = sorted(frame_files, key=lambda x: int(''.join(filter(str.isdigit, x))))
print(frame_files)

# Read the first frame to get width and height
first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
height, width, _ = first_frame.shape

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also try 'XVID' for .avi
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Write frames to the video
for file in frame_files:
    frame = cv2.imread(os.path.join(frames_folder, file))
    video_writer.write(frame)

video_writer.release()
print(f"Video saved as {output_video}")