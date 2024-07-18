import cv2
import numpy as np
from ultralytics import YOLO
from roi import *


def main():
    video_path = 'images/restaurant.mp4'
    model_path = "../YOLOv8/Yolo-Weights/yolov8x.pt"
    output_path = 'output_image_with_boxes.jpg'

    # Extract first frame
    video_processor = VideoProcessor(video_path)
    first_frame = video_processor.extract_first_frame()

    if first_frame is not None:
        # Detect objects
        object_detector = RegionsDetector(model_path)
        results = object_detector.detect_objects(first_frame, classes=[60], conf=0.4)
        
        # Get expanded boxes
        dining_table_boxes = object_detector.get_expanded_boxes(results, 45, first_frame.shape)
        
        # Draw boxes and save image
        ImageProcessor.draw_boxes(first_frame, dining_table_boxes)
        ImageProcessor.save_image(first_frame, output_path)
        
        print("Image with boxes saved successfully.")
    else:
        print("Failed to extract first frame.")

if __name__ == "__main__":
    main()