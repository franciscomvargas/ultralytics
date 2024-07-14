from ultralytics import YOLO
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(curr_dir, '6th_tut', 'dance4pose.mp4')

# Load pretrained YOLOv8 model (nano size)
model = YOLO('yolov8m.pt')

# Run inference on local source
results = model.track(source=filepath, show=True, tracker='bytetrack.yaml')

# # Run inference on live webcam
# results = model.track(source=0, show=True, tracker='bytetrack.yaml')