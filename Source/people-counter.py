import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# COCO Classes
classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']

# Setup video
vid = cv2.VideoCapture("../Videos/mall.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_video.mp4",fourcc,30,(1920,1080))
# Setup Yolov8
model = YOLO("../weights/yolov8n.pt")
# Tracker
tracker = Sort(max_age=20, min_hits=2)
# Mask
mask = cv2.imread("../Images/people-mask.png")
# Line limits
limit_down = [900,960,1150,850] # x1,y1,x2,y2
limit_up = [220,350,460,310]
# Counter
counter_down = []
counter_up = []
# Graphics
counter_image = cv2.imread("../Images/people-counter.png",cv2.IMREAD_UNCHANGED)
while True:
    detections = np.empty((0, 5))
    ret,frame = vid.read()
    if not ret:
        break
    # Masking
    frame_masked = cv2.bitwise_and(frame,mask)
    # Detector Line
    cv2.line(frame,(limit_up[0],limit_up[1]),
             (limit_up[2],limit_up[3]),(0,0,255),4)
    cv2.line(frame, (limit_down[0], limit_down[1]),
             (limit_down[2], limit_down[3]), (0, 0, 255), 4)
    # Adding Graphic
    cvzone.overlayPNG(frame,counter_image,[1440,450])
    cv2.putText(frame,f"{len(counter_up)}",(1575,540),cv2.FONT_HERSHEY_PLAIN,6,(0,0,255),5)
    cv2.putText(frame, f"{len(counter_down)}", (1825, 540), cv2.FONT_HERSHEY_PLAIN, 6, (0, 0, 255), 5)
    # Detect Section
    results = model(frame_masked,stream=True)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Bounding Box
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            w,h = x2-x1,y2-y1
            # Confidence
            conf = int(box.conf[0]*100)/100
            # Class Name
            cls = int(box.cls[0])


            class_name = classes[cls]
            if class_name == "person" and conf >.35:

                bounding_box = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,bounding_box))

    #Tracking Section
    result_tracked = tracker.update(detections)
    for result in result_tracked:
        x1,y1,x2,y2,ID = map(int,result)
        w,h = x2-x1,y2-y1
        cvzone.cornerRect(frame, (x1, y1, w, h))
        cvzone.putTextRect(frame, f"ID: {ID}", (max(0,x1), max(20,y1-20)), scale=1, thickness=1)
        # Center Box
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        # Count UP
        if (limit_up[0]<cx<limit_up[2]) and (limit_up[1]-20<cy<limit_up[1]+20) and counter_up.count(ID)==0:
            counter_up.append(ID)
            cv2.line(frame, (limit_up[0], limit_up[1]),
                     (limit_up[2], limit_up[3]), (0, 255,0), 4)

        # Count Down
        if (limit_down[0] < cx < limit_down[2]) and (limit_down[3]-30 < cy < limit_down[3]+30) and counter_down.count(ID)==0:
            counter_down.append(ID)
            cv2.line(frame, (limit_down[0], limit_down[1]),
                     (limit_down[2], limit_down[3]), (0, 255, 0), 4)

    # Saving Video
    out.write(frame)
    # Display Section
    cv2.imshow("video",frame)
    if cv2.waitKey(1) == 27:
        break

out.release()
vid.release()
cv2.destroyAllWindows()
