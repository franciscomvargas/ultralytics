from ultralytics import YOLO
import sys
sys.path.append('../')

# Load Pre-trained YOLOv8n model
model = YOLO('yolov8n.pt')

# Run inference on live webcam
# results = model(source=0, show=True, conf=0.4, save=False) # Generator of Results Objects

# Run inference on local file
results = model(source='bus.jpg', show=True, conf=0.4, save=False) # Generator of Results Objects

