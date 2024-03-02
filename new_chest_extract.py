from ultralytics import YOLO
import cv2
import numpy as np


model_face = YOLO('yolov8n-face.pt')
model_pose = YOLO('yolov8n-pose.pt')

def draw_rectangle(frame, coords):
    coords = np.array(coords, dtype=np.int32)
    coords = coords.flatten()
    #cv2.imshow("chest", frame)
    #cv2.waitKey(0)

    return coords

def distance(p1,p2):
    return np.sqrt((p2[0]-p1[0])**2-(p2[1]-p1[1])**2)

def isocles(x,y):
    base_length=distance(x,y)
    equal_side_length=base_length/1.5   #Length of two equal sides
    center_x=(x[0]+y[0])/2
    center_y=(x[1]+y[1])/2

    half_base=base_length/2
    half_equal_side=equal_side_length/2

    top_vertex=np.array([center_x,center_y+half_equal_side])
    return top_vertex
    

def equilateral(x, y):
        v_x = (x[0] + y[0] + np.sqrt(3) * (x[1] - y[1])) / 2  # This computes the `x coordinate` of the third vertex.
        v_y = (x[1] + y[1] - np.sqrt(3) * (x[0] - y[0])) / 2
        if v_y < y[1]:
            v_y = (x[1] + y[1] + np.sqrt(3) * (x[0] - y[0])) / 2

        z = np.array([v_x, v_y])  # This is point z, the third vertex.
        return z

def draw_rectangle_1(frame, center_x, center_y, width, height):
        # Calculate the top-left and bottom-right coordinates of the rectangle
        
        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)
        
        
        if width / 2 != width//2:
            x2 += 1
        if height / 2 != height//2:
            y2 += 1
        
        # If coords are outside frame, move center accordingly
        if x1 < 0:
            print("Moving Center X")
            center_x = center_x - x1
            x2 -= x1
        if y1 < 0:
            print("Moving Center Y")
            center_y = center_y - y1
            y2 -= y1
        if x2 > frame.shape[1]:
            print("Moving Center X")
            center_x = center_x - (x2 - frame.shape[1])
            x2 -= (x2 - frame.shape[1])
        if y2 > frame.shape[0]:
            print(y2)
            print("Moving Center Y")
            center_y = center_y - (y2 - frame.shape[0])
            y1 -= (y2 - frame.shape[0])
        
        # Sanitise the coordinates to ensure they are within the frame
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        chest_roi = x1, y1,x2, y2
        return chest_roi

def chest(frame, resX=-1, resY=-1):

    # Predict faces
    results_face = model_face.predict(frame)

    # Predict poses
    results_pose = model_pose.predict(frame)

    for r_1 in results_pose:
        p_1 = r_1.keypoints.xy.to('cpu').numpy()[0][6]
        p_2 = r_1.keypoints.xy.to('cpu').numpy()[0][5]
        p_4 = r_1.keypoints.xy.to('cpu').numpy()[0][11]
        p_5 = r_1.keypoints.xy.to('cpu').numpy()[0][12]
        print(f"P1:{p_1}")
        print(f"P2:{p_2}")
    for r in results_face:
        result = r.boxes.xywh.to('cpu').numpy() 
        print(result)
        print(len(result))
    
    print(f"Length of result:{len(result)}")
    if len(result) == 0:
        p_5[1] = (p_5[1] + p_2[1]) / 1.9
        p_4[1] = (p_4[1] + p_1[1]) / 1.9
        p_1[1] = p_1[1]
        p_2[1] = p_2[1] 
        
        centX = (p_1[0] + p_2[0]) // 2
        centY = (p_1[1] + p_2[1]) // 2
        if resX != -1:
            w = resX
            h = resY
            return draw_rectangle_1(frame, centX, centY, w, h)
        coordinates = [p_1, p_2, p_4, p_5]
        
        w = resX
        h = resY
        return draw_rectangle_1(frame, centX, centY, w, h)
    else:
        point1 = p_1
        point2 = p_2
        # p_3 = equilateral(point1, point2)
        p_3 = isocles(point1, point2)

        if resX != -1:
            w = resX
            h = resY
        else:
            # w = result[0][2] * 2
            # h = result[0][3] * 2
            w = result[0][2] * 2.15
            h = result[0][3] * 2.5/1.75
        x = p_3[0]
        y = p_3[1]
        
        # Draw rectangle on the frame
        return draw_rectangle_1(frame, x, y, w, h)

def body(frame,first_box_width=-1,first_box_height=-1):
    # Predict poses
    results_pose = model_pose.predict(frame, device="CPU")
    for i, r_1 in enumerate(results_pose):
        body_box = r_1.boxes.xyxy.to('cpu').numpy()
        p_1 = r_1.keypoints.xy.to('cpu').numpy()[0][6]
        p_2 = r_1.keypoints.xy.to('cpu').numpy()[0][11]

    # Initialize variables to store the size of the first detected bounding box
    if first_box_width==-1 and first_box_height==-1:

            body_box[0][1]=p_1[1]
            first_box_width = body_box[0][2] - body_box[0][0]
            first_box_height = body_box[0][3] - body_box[0][1]

            if first_box_width % 4 != 0:
                first_box_width -= first_box_width % 4
            if first_box_height % 4 != 0:
                first_box_height -= first_box_height % 4
            
            center_x = (body_box[0][0] + body_box[0][2]) // 2
            center_y = (body_box[0][1] + body_box[0][3]) // 2
            if center_x %2 != 0:
                center_x += 1
            if center_y %2 != 0:
                center_y += 1
                
            new_x1 = int(center_x - first_box_width / 2)
            new_y1 = int(center_y - first_box_height / 2)
            new_x2 = int(center_x + first_box_width / 2)
            new_y2 = int(center_y + first_box_height / 2)

            body_box[0][0] = new_x1
            body_box[0][1] = new_y1
            body_box[0][2] = new_x2
            body_box[0][3] = new_y2

        # Draw the bounding box on the frame
            # cv2.rectangle(frame, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 1)
            # cv2.imshow("",frame)
            # cv2.waitKey(0)
    else:
        center_x = (body_box[0][0] + body_box[0][2]) // 2
        center_y = (body_box[0][1] + body_box[0][3]) // 2
        
        # Calculate the new bounding box coordinates based on the size of the first detection
        new_x1 = int(center_x - first_box_width / 2)
        new_y1 = int(center_y - first_box_height / 2)
        new_x2 = int(center_x + first_box_width / 2)
        new_y2 = int(center_y + first_box_height / 2)
        
        # If coords are outside frame, move center accordingly
        if new_x1 < 0:
            print("Moving Center X")
            center_x = center_x - new_x1
            new_x2 -= new_x1
        if new_y1 < 0:
            print("Moving Center Y")
            center_y = center_y - new_y1
            new_y2 -= new_y1
        if new_x2 > frame.shape[1]:
            print(f"Moving Center X from {center_x} to {center_x - (new_x2 - frame.shape[1])}")
            center_x = center_x - (new_x2 - frame.shape[1])
            new_x1 -= (new_x2 - frame.shape[1])
        if new_y2 > frame.shape[0]:
            print(new_y2)
            print("Moving Center Y")
            center_y = center_y - (new_y2 - frame.shape[0])
            new_y1 -= (new_y2 - frame.shape[0])
        
        print(f"new_x1: {new_x1}, new_y1: {new_y1}, new_x2: {new_x2}, new_y2: {new_y2}")

    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(frame.shape[1], new_x2)
    new_y2 = min(frame.shape[0], new_y2)
    
    return new_x1, new_y1, new_x2, new_y2

# Example usage:
# result = body(frame)
# print(result)



# Example usage:
# result = body(frame)
# print(result)

if __name__ == "__main__":
    image=cv2.imread("new frames/frame_0000.png")
    body(image)