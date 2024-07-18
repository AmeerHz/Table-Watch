import cv2
import torch
import math
from ultralytics import YOLO
import cvzone

# Check if CUDA is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the video
cap = cv2.VideoCapture("images/restaurant.mp4")

# Load the YOLO model
model = YOLO("../YOLOv8/Yolo-Weights/yolov8x.pt").to(device)

# Define the class names
classNames = ["person", "chair", "diningtable"]  # Adjust class names to match your model's output

# Loop through each frame of the video
while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    # Convert the image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get the results from the model
    results = model(img_rgb,classes=[0,39])

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

    # Display the image with bounding boxes
    cv2.imshow('Restaurant', img)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
