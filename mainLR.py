import cv2
import torch
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression

def detect_line_using_dbscan(points, eps=50, min_samples=3):
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

    # Find the cluster with the most points (largest cluster)
    largest_cluster = max(set(clustering.labels_), key=list(clustering.labels_).count)

    # Get the points belonging to the largest cluster
    line_points = points[clustering.labels_ == largest_cluster]

    return line_points

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture each frame from the webcam
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (640, 360))  # Change the dimensions as needed

    # Perform object detection
    results = model(frame)

    # Get the detected bottle coordinates (center points)
    bottles_left = []
    bottles_right = []
    frame_center_x = frame.shape[1] / 2

    for obj in results.xyxy[0]:
        if results.names[int(obj[5])] == 'bottle':
            x_center = (obj[0] + obj[2]) / 2
            y_center = (obj[1] + obj[3]) / 2
            if x_center < frame_center_x:
                bottles_left.append([x_center.item(), y_center.item()])
            else:
                bottles_right.append([y_center.item(), x_center.item()]) # Swap x and y for right side line fitting

    bottles_left = np.array(bottles_left)
    bottles_right = np.array(bottles_right)

    # Detect a row of bottles as a line using DBSCAN clustering and fit a line
    for bottles, is_right_side in [(bottles_left, False), (bottles_right, True)]:
        if len(bottles) > 2:
            line_points = detect_line_using_dbscan(bottles)

            # Fit a line to the points
            line_model = LinearRegression().fit(line_points[:, 0].reshape(-1, 1), line_points[:, 1])

            # Get the start and end points of the line
            start_point = (int(line_points[:, 0].min()), int(line_model.predict(line_points[:, 0].min().reshape(-1, 1))))
            end_point = (int(line_points[:, 0].max()), int(line_model.predict(line_points[:, 0].max().reshape(-1, 1))))

            # Swap back x and y for right side line points
            if is_right_side:
                start_point, end_point = (start_point[1], start_point[0]), (end_point[1], end_point[0])

            # Draw the line
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

            # Draw circles around the bottles forming a line
            for point in line_points:
                cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

    # Show the frame with detected objects and line
    cv2.imshow('YOLOv5 Webcam Object Detection', frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
