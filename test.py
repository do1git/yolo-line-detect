import cv2
import torch
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression

def fit_line(points):
    # Fit a line to the points
    line_model = LinearRegression().fit(points[:, 0].reshape(-1, 1), points[:, 1])

    # Get the start and end points of the line
    start_point = (int(points[:, 0].min()), int(line_model.predict(points[:, 0].min().reshape(-1, 1))))
    end_point = (int(points[:, 0].max()), int(line_model.predict(points[:, 0].max().reshape(-1, 1))))

    return start_point, end_point

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
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

        for label in unique_labels:
            if label != -1:  # Ignore the -1 label, which represents noise
                line_points = bottles[clustering.labels_ == label]

                # Fit a line to the points
                start_point, end_point = fit_line(line_points)

                # Draw the line
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

                # Draw circles around the bottles forming a line
                for point in line_points:
                    cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

    # Show the frame with detected objects and lines
    cv2.imshow('YOLOv5 Webcam Object Detection', frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()