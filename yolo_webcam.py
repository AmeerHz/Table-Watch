import cvzone
from ultralytics import YOLO
import cv2
import math
import time

# Load the video
cap = cv2.VideoCapture("images/restaurant.mp4")

# Load the model
model = YOLO("../YOLOv8/Yolo-Weights/yolov8x.pt")

# Define class names
classNames = ["person", "chair", "dining table"]

# Initialize timing variables
last_processed_time = 0
process_interval = 10

# Initialize table status and detections log
table_status = {}
detections_log = []

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from webcam or video.")
            break

        current_time = time.time()
        if current_time - last_processed_time > process_interval:
            results = model(img, stream=True)
            detection_counts = {"person": 0, "dining table": 0, "chair": 0}

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(img, (x1, y1, w, h))
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])

                    # Debugging prints
                    print(f"Detected class ID: {cls}, Confidence: {conf}, Bounding box: {(x1, y1, x2, y2)}")

                    if cls < len(classNames):
                        label = classNames[cls]
                        print(f"Detected label: {label}")
                        if label in detection_counts:
                            detection_counts[label] += 1
                        cvzone.putTextRect(img, f'{label} {conf}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1)

            print(f"Detection counts: {detection_counts}")

            occupied_tables = 0
            if detection_counts["dining table"] > 0:
                for i in range(detection_counts["dining table"]):
                    table_number = i + 1
                    if detection_counts["person"] > 0:
                        table_status[table_number] = "مشغولة"
                        occupied_tables += 1
                    else:
                        table_status[table_number] = "فارغة"

                print(f"Table status: {table_status}")

                if occupied_tables > 0:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
                    detections_log.append((timestamp, table_status.copy()))

            last_processed_time = current_time

        cv2.imshow("Webcam", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()

    for log in detections_log:
        print(f"في الوقت {log[0]} كانت حالة الطاولات كالتالي: {log[1]}")
