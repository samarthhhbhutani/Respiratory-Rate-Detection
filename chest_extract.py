from ultralytics import YOLO
import numpy as np
import cv2
model = YOLO('yolov8n-face.pt')  # load a pretrained model (recommended for training)
# image_path="frames_5M/frame_0415.png"

model_1 = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)

# results=model.predict(image_path)
# results_1=model_1.predict(image_path)
# for r_1 in results_1:
#     print(r_1.keypoints.xy.numpy())
#     p_1=r_1.keypoints.xy.numpy()[0][6]
#     p_2=r_1.keypoints.xy.numpy()[0][5]

# for r in results:
#     result=r.boxes.xywh.numpy()
import numpy as np


def equilateral(x, y):
    v_x = (x[0] + y[0] + np.sqrt(3) * (x[1] - y[1])) / 2  # This computes the `x coordinate` of the third vertex.
    v_y = (x[1] + y[1] - np.sqrt(3) * (x[0] - y[0])) / 2  # This computes the 'y coordinate' of the third vertex.
    z = np.array([v_x, v_y])  # This is point z, the third vertex.
    return z
def draw_rectangle(image, center_x, center_y, width, height):
    # Calculate the top-left and bottom-right coordinates of the rectangle
    x1 = int(center_x - width / 2)
    y1 = int(center_y - height / 2)
    x2 = int(center_x + width / 2)
    y2 = int(center_y + height / 2)

    # Draw the rectangle on the image
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # (0, 255, 0) is the color in BGR format, here it's green
    return image

from ultralytics import YOLO
import cv2
import numpy as np
def draw_rectangle(img, coords):
    # Convert coordinates to integers
    coords = np.array(coords, dtype=np.int32)

    # Reshape the coordinates to a 2D array
    coords = coords.reshape((-1, 1, 2))
    # Draw the rectangle on the image
    cv2.polylines(img, [coords], isClosed=True, color=(0, 255, 0), thickness=1)

    # Display the image with the rectangle

    return img
def equilateral(x, y):
    v_x = (x[0] + y[0] + np.sqrt(3) * (x[1] - y[1])) / 2  # This computes the `x coordinate` of the third vertex.
    v_y = (x[1] + y[1] - np.sqrt(3) * (x[0] - y[0])) / 2
    if v_y<y[1]:
        v_y = (x[1] + y[1] + np.sqrt(3) * (x[0] - y[0])) / 2

    z = np.array([v_x, v_y])  # This is point z, the third vertex.
    return z
def draw_rectangle_1(image, center_x, center_y, width, height):
    # Calculate the top-left and bottom-right coordinates of the rectangle
    x1 = int(center_x - width / 2)
    y1 = int(center_y - height / 2)
    x2 = int(center_x + width / 2)
    y2 = int(center_y + height / 2)

    # Draw the rectangle on the image
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # (0, 255, 0) is the color in BGR format, here it's green
    return image
# Load YOLO models
model_face = YOLO('yolov8n-face.pt')
model_pose = YOLO('yolov8n-pose.pt')

# Open video file
video_path = "../P1_Virtual_Dataset/extracted_cases/case4_10sec.mp4"
cap = cv2.VideoCapture(video_path)
output_video_path = "output_video_10.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Predict faces
    results_face = model_face.predict(frame,device='CPU')

    # Predict poses
    results_pose = model_pose.predict(frame,device="CPU")

    for r_1 in results_pose:
        print(r_1.keypoints.xy.numpy())
        p_1 = r_1.keypoints.xy.numpy()[0][6]
        p_2 = r_1.keypoints.xy.numpy()[0][5]
        p_4 = r_1.keypoints.xy.numpy()[0][11]
        p_5 = r_1.keypoints.xy.numpy()[0][12]
    for r in results_face:
        result = r.boxes.xywh.numpy()
        print(result)
        print(len(result))

    if len(result)==0:
        p_5[1] = (p_5[1] + p_2[1]) / 1.9
        p_4[1] = (p_4[1] + p_1[1]) / 1.9
        p_1[1] = p_1[1] * 1.05
        p_2[1] = p_2[1] * 1.05
        coordinates = [p_1, p_2, p_4, p_5]
        draw_rectangle(frame, coordinates)
    else:
     point1 = p_1
     point2 = p_2
     p_3 = equilateral(point1, point2)
     w = result[0][2] * 1.5
     h = result[0][3] * 1.5
     x = p_3[0]
     y = p_3[1]

        # Draw rectangle on the frame
     frame = draw_rectangle_1(frame, x, y, w, h)

    # Display the frame
    cv2.imshow("Video", frame)
    out.write(frame)

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
