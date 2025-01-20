import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
#import time
#frame_delay = 0.2
#confidence_threshold = 0.5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp3/weights/last.pt', force_reload=True)
model2 = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')
#model.classes = [0, 1, 2, 3, 5, 6, 7, 9, 11, 15, 16]
'''  0: person #y
  1: bicycle #y
  2: car #y
  3: motorcycle #y
  5: bus #y
  6: train #y
  7: truck #y
  9: traffic light #y
  11: stop sign #y
  15: cat #y
  16: dog #y '''

#allowed_class_ids = [0, 1, 2, 3, 5, 6, 7, 9, 11, 15, 16]

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    results = model(frame)
    results2 = model2(frame)

    for det in results.xyxy[0]:
      class_id, x_min, y_min, x_max, y_max, conf = det.tolist()

      # Ensure class_id is within the allowed class IDs
      #if int(class_id) in allowed_class_ids and conf >0.25: #and conf > 0.5:
        #class_name = model.names[int(class_id)]
        #print(f"The bounding box coordinates for {class_name} are: "
                #f"Top-Left (x, y) = ({int(x_min)}, {int(y_min)}), "
                #f"Bottom-Right (x, y) = ({int(x_max)}, {int(y_max)})")
          
    #top left is [0,0] (x_min, y_min)
    cv2.imshow('YOLO', np.squeeze(results.render()))
    cv2.imshow('YOLO', np.squeeze(results2.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()