from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the YOLOv8 model for person detection.
        We default to yolov8n.pt (nano) for real-time performance.
        """
        self.model = YOLO(model_path)
        # Class 0 in COCO dataset is 'person'
        self.classes = [0]

    def process_frame(self, frame):
        """
        Run detection on a single frame.
        Returns the original frame with boxes drawn, and a list of bounding boxes.
        """
        results = self.model.predict(source=frame, classes=self.classes, conf=0.3, verbose=False)
        annotated_frame = results[0].plot()
        
        # Extract bounding boxes
        boxes = []
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
            
        return annotated_frame, boxes
