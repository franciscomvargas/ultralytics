from ultralytics import YOLO

# Load pre-trained `custom` YOLOv8m model
model = YOLO('tutorials/4th_tut/ducksModel.pt')

# Run inference on the source
results = model(source=0, show=True, conf=0.4, save=False)