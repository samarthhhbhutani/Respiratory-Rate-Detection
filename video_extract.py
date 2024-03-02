from ultralytics import YOLO
import cv2
import os

def extract_video_name(video_name):
    path_segments=video_name.split('\\')
    vid_name=path_segments[-1]
    video_name=vid_name.split('.')
    vidname=video_name[0]
    return vidname


def create_folder_if_not_exists(folder_name):
    # Check if the folder exists
    if not os.path.exists(folder_name):
        # If not, create the folder
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")



model_face = YOLO('yolov8n-face.pt')
model_pose = YOLO('yolov8n-pose.pt')

if __name__ == "__main__":
    video_path="C:\\Users\\bhuta\\Downloads\\UAV_P02_5m_F_RGB.mp4"
    root_folder="C:\\Users\\bhuta\\OneDrive\\Documents\\Respmon\\Respiratory_monitor-master\\Videos"
    cap = cv2.VideoCapture(video_path)
    print(f"Inside video_extract:{video_path}")
    vidname=extract_video_name(video_path)
    print(f"Vidname:{vidname}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))

    frames=[]
    i=0
    j=0
    flag = False
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Predict faces

        # Predict poses
        if not flag:
            results_face = model_face.predict(frame)
            results_pose = model_pose.predict(frame)

            for r_1 in results_pose:
                result_1 = r_1.boxes.xyxy.to('cpu').numpy()
            for r in results_face:
                result = r.boxes.xyxy.to('cpu').numpy()
            flag = True
        output_video_path = f"{root_folder}/{vidname}_{j}.mp4"
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if flag:
            i+=1
            print(j)
            print(i)
            if i>=900:
                frames.append(frame)
            if i==1800:
                    i=0
                    for frame_1 in frames:
                        out.write(frame_1)
                    frames=[]
                    flag = False
                    print("saved to",output_video_path)
                    j+=1
            

    cap.release()
    cv2.destroyAllWindows()
