from ultralytics import YOLO

# Load Small YOLOv8 Pytorch Model
model = YOLO('yolov8s.pt')

# Export the model ~ https://docs.ultralytics.com/modes/export/
model.export(format='openvino') # creates 'yolov8n_openvino_model/'

# Load the exported OpenVino Model
ov_model = YOLO('yolov8s_openvino_model/')

# Run Inference
results = ov_model(source=0, show=True, save=False)
