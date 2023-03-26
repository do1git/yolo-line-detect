import cv2
import torch
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
import time

def fit_line(points):
    # Fit a line to the points
    line_model = LinearRegression().fit(points[:, 0].reshape(-1, 1), points[:, 1])

    # Get the start and end points of the line
    start_point = (int(points[:, 0].min()), int(line_model.predict(points[:, 0].min().reshape(-1, 1))))
    end_point = (int(points[:, 0].max()), int(line_model.predict(points[:, 0].max().reshape(-1, 1))))

    return start_point, end_point

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Start the webcam
cap = cv2.VideoCapture(0)

max_speed = 10  # Maximum speed (km/h)

while True:
    start_time = time.time()

    # Capture each frame from the webcam
    ret, frame = cap.read()

    # Perform object detection
    results = model(frame)

    # Get the detected bottle coordinates (center points)
    bottles = []
    for obj in results.xyxy[0]:
        if results.names[int(obj[5])] == 'bottle':
            x_center = (obj[0] + obj[2]) / 2
            y_center = (obj[1] + obj[3]) / 2
            bottles.append([x_center.item(), y_center.item()])
    bottles = np.array(bottles)

    # Apply DBSCAN clustering to group bottles based on density
    if len(bottles) > 2:
        clustering = DBSCAN(eps=50, min_samples=3).fit(bottles)
        unique_labels = set(clustering.labels_)

        left_line_points = None
        right_line_points = None

        for label in unique_labels:
            if label != -1:  # Ignore the -1 label, which represents noise
                line_points = bottles[clustering.labels_ == label]

                # Fit a line to the points
                start_point, end_point = fit_line(line_points)

                # Check if it's the left or right line
                if start_point[1] < frame.shape[1] / 2:
                    left_line_points = line_points
                else:
                    right_line_points = line_points

        # Calculate the steering angle
        if left_line_points is not None and right_line_points is not None:
            left_line_center = left_line_points.mean(axis=0)
            right_line_center = right_line_points.mean(axis=0)
            center_point = ((left_line_center + right_line_center) / 2).astype(int)

            # Calculate the angle between the car's current orientation and the center point
            # For simplicity, we assume the car is located at the center bottom of the frame and oriented straight up
            car_position = np.array([frame.shape[1] // 2, frame.shape[0]])
            delta_x = center_point[0] - car_position[0]
            delta_y = car_position[1] - center_point[1]
            steering_angle = np.arctan2(delta_x, delta_y) * 180 / np.pi

            # Set the speed based on the steering angle
            # Assuming a linear relationship between steering angle and speed
            speed = max_speed * (1 - abs(steering_angle) / 90)

            # Display the speed and steering angle on the terminal
            print(f"Speed: {speed:.2f} km/h, Steering Angle: {steering_angle:.2f} degrees")

        # Wait for the next update
        elapsed_time = time.time() - start_time
        time.sleep(max(0, 1 - elapsed_time))