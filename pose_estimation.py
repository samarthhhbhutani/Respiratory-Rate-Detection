from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

model.predict("/home/arnav/P1_Virtual_Dataset/extracted_cases/case7.mp4", save=True)