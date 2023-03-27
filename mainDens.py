import cv2
import torch
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression


# 밀도로 군집화

def fit_line(points):
    # Fit a line to the points
    line_model = LinearRegression().fit(points[:, 0].reshape(-1, 1), points[:, 1]) #선형근사().fit(모든 row의 첫번째 column->x좌표.가공, y좌표가공)

    # Get the start and end points of the line
    start_point = (int(points[:, 0].min()), int(line_model.predict(points[:, 0].min().reshape(-1, 1))))
    end_point = (int(points[:, 0].max()), int(line_model.predict(points[:, 0].max().reshape(-1, 1))))

    return start_point, end_point


# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')  # 깃허브에서 모델 다운로드. (깃허브주소, 어떤거를(인자))

# Start the webcam
cap = cv2.VideoCapture(0)  # 0번째 카메라(기본카메라)사용

while True:  # 무한실행
    # Capture each frame from the webcam
    ret, frame = cap.read()  # 카메라 열였는지에 대한 Bool, 현재프레임

    # Perform object detection
    results = model(frame)  # 로드한 모델에 프레임(현재순간장면)삽입

    # Get the detected bottle coordinates (center points)
    bottles = []
    #    print(f"-->xyxy[0]<--{results.xyxy[0]}")
    # tensor([[1.23934e+03, 4.66973e+02, 1.58346e+03, 7.28873e+02, 3.24482e-01, 6.30000e+01],
    #         [9.05466e+02, 5.92505e+02, 1.15413e+03, 1.07558e+03, 2.77116e-01, 3.90000e+01],
    #         [4.27986e-01, 6.81915e+00, 3.32822e+02, 6.95988e+02, 2.66396e-01, 0.00000e+00]])
    # [x1, y1, x2, y2, confidence, class]
    # [위왼쪽 모서리좌표x,위왼쪽 모서리좌표y, 아래오른쪽 모서리좌표x, 아래오른쪽 모서리좌표y, 확률, 대상]

    # print(f"-->names<--{results.names}")
    # -->names < --{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
    #               8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    #               14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    #               22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
    #               29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    #               35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    #               40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
    #               47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
    #               54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    #               61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
    #               68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    #               75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

    for obj in results.xyxy[0]:  # 인식한 친구들 중에서
        if results.names[int(obj[5])] == 'bottle':  # bottle 이라는 물체가 있으면
            x_center = (obj[0] + obj[2]) / 2  # 중간좌표값 추출
            y_center = (obj[1] + obj[3]) / 2
            cv2.circle(frame, (int(x_center), int(y_center)), radius=5, color=(255, 0, 1), thickness=-1)
            bottles.append([x_center.item(), y_center.item()])
    bottles = np.array(bottles)
    # Apply DBSCAN clustering to group bottles based on density
    if len(bottles) > 2:  # bottle이 2개초과면
        clustering = DBSCAN(eps=150, min_samples=3).fit(bottles)  # 밀도기반 공간클러스터링. 포인트 간의 eps거리와 최소갯수.
        unique_labels = set(clustering.labels_) # DBSCAN으로 군집화 진행
        for label in unique_labels:
            if label != -1:  # Ignore the -1 label, which represents noise # {-1, 0}에서 -1은 노이즈니까 무시
                line_points = bottles[clustering.labels_ == label] # for문 중 해당 점들 찾기
                # Fit a line to the points
                start_point, end_point = fit_line(line_points) # 처음과 끝을 라인화

                # Draw the line
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2) #선 표시

                # Draw circles around the bottles forming a line
                for point in line_points: # 감지된 점들을 빨간색표시
                    cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

    # Show the frame with detected objects and lines
    cv2.imshow('YOLOv5 Webcam water BOTTLE Detection', frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
