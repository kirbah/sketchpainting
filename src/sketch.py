import cv2
import numpy as np
from moviepy import ImageSequenceClip

# Parameters
input_image = 'source/input_v2_small.jpg'          # Path to your input image
output_video = 'output/drawing_animation_v2.mp4'
frame_rate = 30                    # Frames per second

# Load the image and convert to grayscale
img = cv2.imread(input_image)
if img is None:
    raise ValueError("Could not load image. Check the path.")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use Canny edge detection to extract edges
edges = cv2.Canny(gray, threshold1=100, threshold2=200)

# Find contours from the edges
contours, hierarchy = cv2.findContours(
    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Optionally, sort contours by area (largest first)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

frames = []  # To store each frame of the animation

# Create a blank white canvas (same size as the original image)
canvas = np.ones_like(img) * 255

# For each contour, draw the stroke incrementally
for cnt in contours:
    # Simplify the contour shape if needed (remove redundant points)
    cnt = cv2.approxPolyDP(cnt, epsilon=1.0, closed=False)

    # Reshape for convenience
    cnt = cnt.squeeze()
    if cnt.ndim == 1:  # If only one point, skip it
        continue

    num_points = len(cnt)
    # Gradually draw the contour by adding one point at a time
    for i in range(1, num_points + 1):
        # Make a copy of the current canvas
        canvas_copy = canvas.copy()
        # Get the subset of points for this stroke
        pts = cnt[:i].reshape((-1, 1, 2))
        # Draw the current segment on the copy
        cv2.polylines(canvas_copy, [pts], isClosed=False,
                      color=(0, 0, 0), thickness=2)
        frames.append(canvas_copy)

    # Once the stroke is complete, update the canvas permanently
    cv2.polylines(canvas, [cnt.reshape((-1, 1, 2))],
                  isClosed=False, color=(0, 0, 0), thickness=2)

# Optionally, hold the final image for a few frames at the end of the animation
for _ in range(10):
    frames.append(canvas.copy())

# Convert frames from BGR (OpenCV format) to RGB (MoviePy format)
frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

# Create a video clip from the image sequence
clip = ImageSequenceClip(frames_rgb, fps=frame_rate)
clip.write_videofile(output_video, codec='libx264')

print("Animation video created:", output_video)
