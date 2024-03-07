'''
Author: Tiago Simões
Date: March 2024

This script performs human pose tracking using YOLOv8 model.

It captures video from a file, applies YOLOv8 model to detect 
and track human poses in each frame, and displays the annotated 
frames with bounding boxes and track IDs.
'''

import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

# Set the height of the tracking line (0-100% of bounding box)
track_height = 25
save_status = False
show_status = True
track_status = True
print_track = False

# Capture video from a file
vid_cap_path = "1o_teste_velocidade_baixa.MOV" #"Viena_30s.mp4"
vid_cap = cv2.VideoCapture(vid_cap_path)

# Save tracking to a file
if save_status:
    vid_save_path = "Viena_Tracker_Full" + str(track_height) + ".avi"
    vid_save = cv2.VideoWriter(vid_save_path, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1920, 1080))

# Load the trained YOLOv8 model
model = YOLO("yolov8n-pose.pt")

track_history = defaultdict(lambda: [])


while vid_cap.isOpened():
    success, frame = vid_cap.read()

    if success:
        

        if track_status:
            # Use YOLOv8 to track human pose in the frame
            result = model.track(frame, persist=True)

            # Draw the bounding boxes and track IDs on the frame
            annotated_frame = result[0].plot()
            
            if print_track:
                # Get the bounding boxes and track IDs for all detected objects
                if result[0].boxes.id is not None:
                    boxes = result[0].boxes.xywh.cpu()
                    track_ids = result[0].boxes.id.int().cpu().tolist()


                    for box, track_id in zip(boxes, track_ids):
                        x,y,w,h = box

                        track = track_history[track_id]
                        track.append((float(x), float(y)+(float(h)*(0.5-track_height/100))))

                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 0, 0), thickness=10)   

        else:
             # Use YOLOv8 to identify human pose in the frame
            result = model(frame)

        im = cv2.resize(annotated_frame, (960, 540))

        if save_status:
            vid_save.write(annotated_frame)

        if show_status:
            cv2.imshow("frame", im)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break

vid_cap.release()
if save_status:
    vid_save.release()
cv2.destroyAllWindows()