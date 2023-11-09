import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Capture video from a file
video_path = "Viena_30s.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        result = model(frame)
        im = cv2.resize(result[0].plot(), (960, 540))
        cv2.imshow("frame", im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

#model.predict(source=video_path, show=True, conf=0.25)