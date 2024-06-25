# segmentation overview
import cv2, time
from ultralytics import YOLO
import sys
sys.path.append('../')

model = YOLO('yolov8n-seg.pt')

# Open the video file
# video_path = 'CAR_MOV.mp4'
video_path = 0 # webcam
cap = cv2.VideoCapture(video_path)

# Loop through the video frame
while cap.isOpened():
    # Read a frame from the video
    sucess, frame = cap.read()

    if sucess:
        start = time.perf_counter()
        
        #Run YOLOv8 inference on the frame
        results = model(frame)

        end = time.perf_counter()
        total_time = end - start 
        fps = 1 / total_time

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.putText(annotated_frame, f'FPS: {int(fps)}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) # https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
        cv2.imshow('YOLOv8 Inference', annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture and close the Display Window
cap.release()
cv2.destroyAllWindows()