# Object Detection Libs
import torch, cv2
from time import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Email Libs
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Get Params
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(curr_dir, 'params.yaml')
if not os.path.isfile(params_path):
    import shutil
    shutil.copyfile(os.path.join(curr_dir, 'params_copy.yaml'), params_path)
import yaml, json
with open (params_path, 'r') as fr:
    PARAMS = yaml.safe_load(fr)

if not PARAMS['email']['pass']:
    raise EnvironmentError(f"EDIT: Parameters file `params.yaml` at: {params_path}")

if PARAMS['debug']:
    print('*'*20, 'PARAMETERS', '*'*20)
    print(json.dumps(PARAMS, indent=2))
    print('*'*52)


# Server creation and authentication
SERVER = smtplib.SMTP('smtp.gmail.com', 587)
server_start_res = SERVER.starttls()
if PARAMS['debug']:
    print("TLS server start res:", server_start_res)
SERVER.login(PARAMS['email']['from'], PARAMS['email']['pass'])


# Email Send Function
def send_email(email_from, email_to, object_detected=1):
    """Sends an email notification indicating the number of objects detected; defaults to 1 object."""
    message = MIMEMultipart()
    
    message['From'] = email_from
    message['To'] = email_to
    message['Subject'] = 'Vargas Security System Alert'
    # Add in the message body
    message_body = f'ALERT - {object_detected} objects has been detected!!'
    print("")

    if PARAMS['debug']:
        print(f'[ DEBUG ] EMAIL MESSAGE:\n{message_body}')
        # exit(0)

    message.attach(MIMEText(message_body, 'plain'))

    SERVER.sendmail(email_from, email_to, message.as_string())
    

# Object Detection and Alert Sender
class ObjectDetection:
    def __init__(self, capture_index):
        """Initializes an ObjectDetection instance with a given camera index."""
        self.capture_index = capture_index
        self.email_sent = False

        # model info
        self.model = YOLO('yolov8n.pt')

        # visual info
        self.annotator = None
        self.start_time = 0
        self.end_time = 0

        # device info
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def predict(self, im0):
        """Run prediction using a YOLO model for the input image `im0`."""
        results = self.model(im0, conf=0.9, save = False)
        return results

    def display_fps(self, im0):
        """Displays the FPS on an image `im0` by calculating and overlaying as white text on a black rectangle."""
        self.end_time = time()
        fps = 1 / round(self.end_time - self.start_time, 2)

        text = f'FPS: {int(fps)}'

        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        gap=10
        
        cv2.rectangle(
            im0,
            (20 - gap, 70 - text_size[1] - gap), # x axis
            (20 + text_size[0] + gap, 70 + gap), # y axis
            (255, 255, 255), # rectangle color
            -1, # border thickness
        )

        cv2.putText(im0, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    def plot_bboxes(self, results, im0):
        """
        Plots bounding boxes on an image given detection results; 
        returns annotated image and class IDs.
        """
        class_ids = []
        self.annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names

        for _box, _cls in zip(boxes, clss):
            class_ids.append(_cls)
            self.annotator.box_label(_box, label=names[int(_cls)], color=colors(int(_cls), True))

        return im0, class_ids

    def __call__(self):
        """
        Run object detection on video frames from a camera stream, 
        plotting and showing the results.
        """
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_count = 0
        while True:
            self.start_time = time()
            
            ret, im0 = cap.read()
            assert ret

            results = self.predict(im0)
            im0, class_ids = self.plot_bboxes(results, im0)

            if len(class_ids) > 0 and not self.email_sent:  # Only send email If not sent before
                send_email(PARAMS['email']['from'], PARAMS['email']['to'], len(class_ids))
                self.email_sent = True
            else:
                self.email_sent = False

            self.display_fps(im0)
            cv2.imshow('YOLOv8 Detection', im0)
            frame_count += 1

            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        server.quit()

detector = ObjectDetection(capture_index=0)
detector()