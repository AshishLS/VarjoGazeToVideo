# This script takes a input video, refers the varjo gaze file and creates a output video that has gaze information marked as shades of circle.

# pip install opencv-python pandas
import cv2
import pandas as pd
import numpy as np

def interpolate_colors(num_colors):
    # Define start color (white) and end color (red)
    start_color = np.array([255, 255, 255])  # White
    end_color = np.array([0, 0, 255])        # Red
    
    # Generate interpolated colors
    colors = [((1 - t) * start_color + t * end_color).astype(int) for t in np.linspace(0, 1, num_colors)]
    
    # Return the list of colors
    return [tuple(color) for color in colors]

# Parameters
video_file = 'Case1/varjo_video.mp4'  # Path to your Varjo video
csv_file = 'Case1/varjo_gaze_output.csv'  # Path to the CSV with gaze data
output_video_file = 'output_varjo_video.mp4'  # Path to the output video
gaze_marker_color = (0, 0, 255)  # Red marker (BGR format for OpenCV)
gaze_marker_radius = 20  # Radius of the gaze marker
gaze_marker_thickness = 5  # Thickness of the marker circle

# Load the eye-tracking data
gaze_data = pd.read_csv(csv_file)

# Open the video file
video_capture = cv2.VideoCapture(video_file)
fps = video_capture.get(cv2.CAP_PROP_FPS)
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare the output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out_video = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

# following is eye candy
# Initialize a list to store the last 10 gaze points
gaze_history_number = 50
gaze_history = []
# Colors with decreasing red intensity
red_shades = interpolate_colors(gaze_history_number)
print( red_shades[0])
# Iterate over the frames of the video
frame_idx = 0
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break  # End of the video

    # Synchronize with the gaze data based on frame index and FPS
    timestamp = (frame_idx / fps) * 1000000000 # the timestamps in the csv are in nanoseconds
    
    gaze_row = gaze_data.iloc[(gaze_data['relative_to_video_first_frame_timestamp'] - timestamp).abs().argmin()]  # Find the closest timestamp in the CSV

    # Ensure that gaze_x and gaze_y columns are floats
    # gaze_data['gaze_projected_to_left_view_x'] = pd.to_numeric(gaze_data['gaze_projected_to_left_view_x'], errors='coerce')
    # gaze_data['gaze_projected_to_left_view_y'] = pd.to_numeric(gaze_data['gaze_projected_to_left_view_y'], errors='coerce')

    # Extract normalized gaze coordinates
    # This is required because the varjo cordinates are normalized to -1,1 from center of screen while opencv frame origin
    # is uppper left corner.
    norm_gaze_x = gaze_row['gaze_forward_x']
    norm_gaze_y = gaze_row['gaze_forward_y']


     # Map normalized gaze coordinates to pixel values
    gaze_x = int((norm_gaze_x + 1) / 2 * frame_width)    # x: [-1,1] -> [0, frame_width]
    gaze_y = int((1 - norm_gaze_y) / 2 * frame_height)   # y: [-1,1] -> [0, frame_height] with a flip

    # debug
    print("t={}, gaze_x={}, gaze_y={}".format(timestamp, gaze_x, gaze_y))

    # Draw a circle marker on the frame at the gaze coordinates
    cv2.circle(frame, (gaze_x, gaze_y), gaze_marker_radius, gaze_marker_color, gaze_marker_thickness)

    # Store the current gaze point in the history # Eye Candy
    gaze_history.append((gaze_x, gaze_y))
    # Keep only the last 10 gaze points
    if len(gaze_history) > gaze_history_number:
        gaze_history.pop(0)

    # Draw circles for each of the last 10 gaze points # Eye Candy
    for i, (gx, gy) in enumerate(gaze_history):
        # Access the first color and convert it to a regular Python tuple of integers
        color = tuple(map(int, red_shades[i]))
        cv2.circle(frame, (gx, gy), i + gaze_marker_radius, color, gaze_marker_thickness)

    # Write the frame with the gaze marker to the output video
    out_video.write(frame)

    frame_idx += 1

# Release resources
video_capture.release()
out_video.release()
cv2.destroyAllWindows()

print(f"Output video saved as {output_video_file}")
