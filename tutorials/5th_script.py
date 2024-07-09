import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

import supervision as sv

class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using Device:', self.device)

        self.model = self.load_model()

        self.bounding_box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def load_model(self):
        model = YOLO('yolov8n.pt')  # load pre-trained model
        model.fuse() # Ultralytics Optimizations
        
        return model

    def predict(self, frame):
        results = self.model(frame)[0]

        return results

    def plot_bboxes(self, results, frame):
        # xyxys = []
        # confs = []
        # class_ids = []

        # # Extract detections for person class
        # for res in results:
        #     boxes = res.boxes.cpu().numpy()
        #     # print('Detected boxes:', boxes) # DEBUG boxes object
        #     _xyxys = boxes.xyxy
        #     # print('Detected xyxy:', xyxys) # GET xyxy atribute from boxes
            
        #     # for xyxy in _xyxys:
        #     #     # # > Collect data
        #     #     # confs.append(xyxy[4])
        #     #     # class_ids.append(xyxy[5])

        #     #     # Plot rectangle around detection
        #     #     cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,0), 1)
            
        #     class_id = boxes.cls[0]
        #     conf = boxes.conf[0]
        #     xyxy = boxes.xyxy[0]

        #     if class_id == 0.0:
        #         class_ids.append(class_id)
        #         confs.append(conf)
        #         xyxys.append(xyxy)

        # Setup detections for visualization
        # # Retrieved from tutorial
        # detections = sv.Detections(
        #     xyxy = results[0].boxes.xyxy.cpu().numpy(),
        #     confidence = results[0].boxes.conf.cpu().numpy(),
        #     class_id = results[0].boxes.cls.cpu().numpy(),
        # )
        
        # Retrieved from roboflow docs (https://supervision.roboflow.com/latest/how_to/detect_and_annotate/#display-custom-labels)
        detections = sv.Detections.from_ultralytics(results)

        # Format custom labels
        # # Retrieved from tutorial
        # _labels = [
        #     f'{self.CLASS_NAMES_DICT[_class_id]} {_confidence:0.2f}'
        #         for _class_id, _confidence, _tracker_id
        #             in zip(detections.class_id, detections.confidence, detections.tracker_id)
        # ]
        # # Personal approach
        # _labels = []
        # for detection in detections:
        #     print('*'*80)
        #     try:
        #         _class_id, _confidence, _tracker_id = zip(detection.class_id, detection.confidence, detection.tracker_id)
        #         print('_class_id:', _class_id)
        #         print('_confidence:', _confidence)
        #         print('_tracker_id:', _tracker_id)
        #         _labels.append(f'{self.CLASS_NAMES_DICT[_class_id]} {_confidence:0.2f}')
        #     except:
        #         print('ERROR:', detection)
        #         continue
        # Retrieved from roboflow docs (https://supervision.roboflow.com/latest/how_to/detect_and_annotate/#display-custom-labels)
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(detections['class_name'], detections.confidence)
        ]

        # Annotate and display frame
        # # Retrieved from tutorial
        # frame = self.box_annotator.annotate(scene=frame, detections = detections, labels = _labels)
        # Retrieved from roboflow docs (https://supervision.roboflow.com/latest/how_to/detect_and_annotate/#display-custom-labels)
        annotated_image = self.bounding_box_annotator.annotate(
            scene=frame, detections=detections)
        frame = self.label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels)


        return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            start_time = time()

            ret, frame = cap.read()
            assert ret

            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)

            end_time = time()
            fps = 1 / np.round(end_time-start_time, 2)

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('YOLOv8 Detection', frame)

            if cv2.waitKey(5) & 0xFF == 27: break
        
        cap.release()
        cv2.destroyAllWindows()


# Make it Slide...
detector = ObjectDetection(capture_index=0)
detector()