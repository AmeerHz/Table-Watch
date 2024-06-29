import cvzone
from ultralytics import YOLO
import cv2
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cap = cv2.VideoCapture("images/res.mp4")
model = YOLO("../YOLOv8/Yolo-Weights/yolov8x.pt").to(device)

classNames = ["diningtable", "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", ]

# إنشاء الشكل والمحور لـ matplotlib
plt.ion()
fig, ax = plt.subplots()

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img)

    tables = []
    people = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if cls < len(classNames):
                className = classNames[cls]
                print(f"Detected {className} with confidence {conf} at [{x1}, {y1}, {x2}, {y2}]")

                if className == "diningtable":
                    print(f"Table detected at [{x1}, {y1}, {x2}, {y2}]")
                    tables.append((x1, y1, x2, y2))
                    cvzone.cornerRect(img, (x1, y1, w, h), colorR=(0, 255, 0))
                    cvzone.putTextRect(img, f'Table {conf}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1)
                elif className == "person":
                    people.append((x1, y1, x2, y2))
                    cvzone.cornerRect(img, (x1, y1, w, h), colorR=(255, 0, 0))
                    cvzone.putTextRect(img, f'Person {conf}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1)

    print(f"Detected {len(tables)} tables and {len(people)} people.")

    for (tx1, ty1, tx2, ty2) in tables:
        table_occupied = False
        for (px1, py1, px2, py2) in people:
            if px1 < tx2 and px2 > tx1 and py1 < ty2 and py2 > ty1:
                table_occupied = True
                break

        color = (0, 0, 255) if table_occupied else (0, 255, 0)
        cv2.rectangle(img, (tx1, ty1), (tx2, ty2), color, 2)
        status = "Occupied" if table_occupied else "Vacant"
        cvzone.putTextRect(img, status, (tx1, ty1 - 10), scale=0.7, thickness=1, colorR=color)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # عرض الصورة باستخدام matplotlib
    ax.clear()
    ax.imshow(img)
    ax.set_title("Restaurant")
    plt.draw()
    plt.pause(5)

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
