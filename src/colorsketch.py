import cv2
import numpy as np
from moviepy import ImageSequenceClip

# ------------------- Configuration -------------------
input_image = 'source/input_v3_small.jpg'          # Path to your input image
output_video = 'output/color_drawing_animation_v3_1.mp4'
frame_rate = 30                    # Frames per second
desired_duration = 120             # Desired final video duration in seconds
# -----------------------------------------------------

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


# -----------------------------------------------------
# Adjusted drawing logic:
# First, determine the total number of drawing segments across all contours.
total_segments = 0
for cnt in contours:
    cnt = cv2.approxPolyDP(cnt, epsilon=1.0, closed=False)
    cnt = cnt.squeeze()
    if cnt.ndim == 1 or len(cnt) < 2:
        continue
    total_segments += len(cnt) - 1

# Compute desired total frames for the video and a step factor to capture fewer frames.
desired_total_frames = desired_duration * frame_rate
step_factor = max(1, int(total_segments / desired_total_frames))
print("Total segments:", total_segments, "Step factor:", step_factor)
# -----------------------------------------------------

# For each contour, draw the stroke incrementally in color, capturing frames only every "step_factor" segments.
for cnt in contours:
    cnt = cv2.approxPolyDP(cnt, epsilon=1.0, closed=False)
    cnt = cnt.squeeze()
    if cnt.ndim == 1 or len(cnt) < 2:
        continue  # Skip if contour is not valid

    num_points = len(cnt)

    # Draw the contour incrementally by drawing segments and capturing frames every "step_factor" segments.
    for i in range(1, num_points):
        # Only capture a frame if the segment index is a multiple of step_factor, or if it's the final segment.
        if i % step_factor != 0 and i != num_points - 1:
            # Also update the permanent canvas for consistency
            pt1 = tuple(cnt[i - 1])
            pt2 = tuple(cnt[i])
            mid_x = int((pt1[0] + pt2[0]) / 2)
            mid_y = int((pt1[1] + pt2[1]) / 2)
            color = get_color((mid_x, mid_y))
            cv2.line(canvas, pt1, pt2, color, thickness=2)
            continue

        # Create a copy of the current canvas to record the incremental stroke
        canvas_copy = canvas.copy()
        for j in range(1, i + 1):
            pt1 = tuple(cnt[j - 1])
            pt2 = tuple(cnt[j])
            mid_x = int((pt1[0] + pt2[0]) / 2)
            mid_y = int((pt1[1] + pt2[1]) / 2)
            color = get_color((mid_x, mid_y))
            cv2.line(canvas_copy, pt1, pt2, color, thickness=1)
        frames.append(canvas_copy)

        # Also update the permanent canvas with a thicker line
        pt1 = tuple(cnt[i - 1])
        pt2 = tuple(cnt[i])
        mid_x = int((pt1[0] + pt2[0]) / 2)
        mid_y = int((pt1[1] + pt2[1]) / 2)
        color = get_color((mid_x, mid_y))
        cv2.line(canvas, pt1, pt2, color, thickness=2)

# Optionally, hold the final image for a few extra frames at the end of the animation
for _ in range(10):
    frames.append(canvas.copy())

# Convert frames from BGR (OpenCV format) to RGB (MoviePy format)
frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

# Create a video clip from the image sequence with constant 30 fps.
clip = ImageSequenceClip(frames_rgb, fps=frame_rate)
clip.write_videofile(output_video, codec='libx264')

print("Color drawing animation video created:", output_video)
