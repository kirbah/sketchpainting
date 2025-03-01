import cv2
import numpy as np
from moviepy import ImageSequenceClip

# Parameters
input_image = 'source/input_v2_small.jpg'          # Path to your input image
output_video = 'output/color_drawing_animation_v2.mp4'
frame_rate = 30                    # Frames per second

# Load the original image (color) and create a grayscale copy for edge detection
img = cv2.imread(input_image)
if img is None:
    raise ValueError("Could not load image. Check the path.")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use Canny edge detection to extract edges
edges = cv2.Canny(gray, threshold1=100, threshold2=200)

# Find contours from the edges
contours, _ = cv2.findContours(
    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Optionally, sort contours by area (largest first)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

frames = []  # To store each frame of the animation

# Create a blank white canvas (same size as the original image)
canvas = np.ones_like(img) * 255

# Function to get color from the original image at a given point


def get_color(point):
    x, y = point
    # Ensure the point is within bounds
    h, w, _ = img.shape
    x = np.clip(x, 0, w-1)
    y = np.clip(y, 0, h-1)
    return tuple(int(c) for c in img[y, x])  # Note: OpenCV uses (y, x)


# For each contour, draw the stroke incrementally in color
for cnt in contours:
    # Optionally simplify the contour
    cnt = cv2.approxPolyDP(cnt, epsilon=1.0, closed=False)
    cnt = cnt.squeeze()
    if cnt.ndim == 1 or len(cnt) < 2:
        continue  # Skip if contour is not valid

    num_points = len(cnt)

    # Draw the contour incrementally by drawing each segment one by one
    for i in range(1, num_points):
        # Create a copy of the current canvas to record the incremental stroke
        canvas_copy = canvas.copy()
        # Draw all segments up to the current point
        for j in range(1, i + 1):
            pt1 = tuple(cnt[j - 1])
            pt2 = tuple(cnt[j])
            # Compute the midpoint of the segment to sample color
            mid_x = int((pt1[0] + pt2[0]) / 2)
            mid_y = int((pt1[1] + pt2[1]) / 2)
            color = get_color((mid_x, mid_y))
            cv2.line(canvas_copy, pt1, pt2, color, thickness=1)
        frames.append(canvas_copy)

    # Once the stroke is complete, update the permanent canvas
    for j in range(1, num_points):
        pt1 = tuple(cnt[j - 1])
        pt2 = tuple(cnt[j])
        mid_x = int((pt1[0] + pt2[0]) / 2)
        mid_y = int((pt1[1] + pt2[1]) / 2)
        color = get_color((mid_x, mid_y))
        cv2.line(canvas, pt1, pt2, color, thickness=2)

# Optionally, hold the final image for a few extra frames at the end of the animation
for _ in range(10):
    frames.append(canvas.copy())

# Convert frames from BGR (OpenCV format) to RGB (MoviePy format)
frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

# Create a video clip from the image sequence
clip = ImageSequenceClip(frames_rgb, fps=frame_rate)
clip.write_videofile(output_video, codec='libx264')

print("Color drawing animation video created:", output_video)
