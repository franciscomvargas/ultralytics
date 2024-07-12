from ultralytics import YOLO
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(curr_dir, 'dance4pose.mp4')

# Load pre-trained YOLOV8 model
model = YOLO('yolov8s-pose.pt')

# # Run Inference on source file
# results = model(source=filepath, show=True, conf=.3, save=False)

# Run Inference on live webcam
results = model(source=0, show=True, conf=.3, save=False)
