import cv2
import numpy as np
from ultralytics import YOLO



class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path

    def extract_first_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return None

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Error: Could not read frame.")
            return None

        return frame

class RegionsDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_objects(self, image, classes, conf):
        return self.model(image, classes=classes, show=True, conf=conf)

    @staticmethod
    def expand_box(box, expand_by, image_shape):
        x1, y1, x2, y2 = box
        height, width, _ = image_shape
        
        x1 = max(0, x1 - expand_by)
        y1 = max(0, y1 - expand_by)
        x2 = min(width, x2 + expand_by)
        y2 = min(height, y2 + expand_by)
        
        return [int(x1), int(y1), int(x2), int(y2)]

    def get_expanded_boxes(self, results, expand_by, image_shape):
        expanded_boxes = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            xyxys = boxes.xyxy
            for box in xyxys:
                expanded_box = self.expand_box(box, expand_by, image_shape)
                expanded_boxes.append(expanded_box)
        return expanded_boxes

class ImageProcessor:
    @staticmethod
    def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    @staticmethod
    def save_image(image, output_path):
        cv2.imwrite(output_path, image)

