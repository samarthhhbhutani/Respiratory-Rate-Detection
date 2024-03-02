from collections import deque

import cv2
import numpy as np
from matplotlib import pyplot as plt
from transforms import butter_lowpass_filter, float_to_uint8, uint8_to_float
import peakutils
from ultralytics import YOLO
from new_chest_extract import body, chest

# def remove_features_on_black(frame, features):
#     print(f"SHAPE OF FRAME:{frame.shape}")
#     if len(frame.shape) == 3:  # Check if the image has 3 channels
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     elif len(frame.shape) == 2:  # If already grayscale, use the frame directly
#         gray = frame
#     else:
#         raise ValueError("Unsupported image format. Expected either 3-channel BGR or grayscale image.")

#     # Threshold the frame to identify black regions
#     _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
#     # Invert the thresholded image
#     thresh = cv2.bitwise_not(thresh)
    
#     # Find contours in the thresholded image
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     frame_with_contours = cv2.drawContours(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), contours, -1, (0, 0, 255), 2)
#     print("Frame w contours displaying")
#     cv2.imshow('Frame with Contours', frame_with_contours)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

    
    # Filter out features that fall within black regions
    # filtered_features = []
    # for feature in features:
    #     x, y = feature.ravel()
    #     # Check if the feature falls within any contour
    #     if all(cv2.pointPolygonTest(cnt, (x, y), False) != 1 for cnt in contours):
    #         filtered_features.append(feature)
    
    # return np.array(filtered_features)


# Example usage



def draw_ROIs(frame, chest_coords, body_coords):
    frame_with_ROIs = frame.copy()
    # Draw rectangle around chest ROI
    cv2.rectangle(frame_with_ROIs, (chest_coords[0], chest_coords[1]), (chest_coords[2], chest_coords[3]), (0, 255, 0), 2)
    # Draw rectangle around body ROI
    cv2.rectangle(frame_with_ROIs, (body_coords[0], body_coords[1]), (body_coords[2], body_coords[3]), (0, 0, 255), 2)
    return frame_with_ROIs


def find_peaks(filtered_data, peak_minimum_sample_distance, t):

    print(f"Max of filtered:{np.max(filtered_data)}")
    gaussian_cutoff = 15.0
    #gaussian_cutoff = 0.05*np.max(filtered_data)
    width = peak_minimum_sample_distance
    print(width)
    #TODO: Change threshold to be a percentage of the max value
    indices = peakutils.indexes(filtered_data, min_dist=width, thres=0.1)

    final_idxs = []
    fits = []
    for idx in indices:
        w = width
        if idx - width < 0:
            w = idx
        if idx + w > len(t):
            w = len(t) - idx
        ti = np.array(t)[idx - w:idx + w]
        datai = np.array(filtered_data)[idx - w:idx + w]
        try:
            params = peakutils.gaussian_fit(ti, datai, center_only=False)
            y = [peakutils.gaussian(x, *params) for x in ti]
            # ax.plot(ti, y)
            ssr = np.sum(np.power(np.subtract(y, datai), 2.0))
            sst = np.sum(np.power(np.subtract(y, datai), 2.0))
            r2 = 1 - (ssr / sst)
            fits.append(r2)
            if params[2] < gaussian_cutoff:
                final_idxs.append(idx)
        except RuntimeError:
            pass
    return final_idxs, fits



def track_motion(ROI, frame_count,pick_random=False):
    motion_data = deque()
    estimated_motion = deque()
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=((cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT), 10, 0.03),
    )
    previous_frame = None
    for frame in ROI:
        if previous_frame is None:

            previous_frame = float_to_uint8(frame)
            frame = float_to_uint8(frame)

            if not pick_random and frame_count % 10 == 0:
                features = cv2.goodFeaturesToTrack(frame, 20, 0.01, 4, blockSize=5)
            if not pick_random:
                features = cv2.goodFeaturesToTrack(frame, 20, 0.01, 4, blockSize=5)
            else:
                #Picking 10 random points as features
                features = np.zeros((100, 1, 2), dtype='float32')
                features[:, :, 0] = np.random.randint(0, frame.shape[1], size=(100, 1))
                features[:, :, 1] = np.random.randint(0, frame.shape[0], size=(100, 1))
            # features = remove_features_on_black(frame, features)  # Remove features on black background
            print(features.shape)
            # features = np.append(features,[[np.array([29.0,15.0], dtype='float32')]], axis=0)
            # features = np.array([29,15], dtype='float32').reshape(-1,1,2)
            print(f"Features: {features}")
            continue
        frame = float_to_uint8(frame)
        (p1, st, _) = cv2.calcOpticalFlowPyrLK(previous_frame, frame, features, None, **lk_params)

        good_new = p1[(st == 1)]
        good_old = features[(st == 1)]

        previous_frame = frame.copy()
        features = good_new.reshape((-1), 1, 2)

        raw_motion = list(np.mean(good_new - good_old, axis=0))
        motion_data.append(raw_motion)

        if len(motion_data) >= 2:
            x, y = np.transpose(motion_data)
            coords = np.vstack([x, y])
            cov_mat = np.cov(coords)
            eig_vals, eig_vecs = np.linalg.eig(cov_mat)
            sort_indices = np.argsort(eig_vals)[::-1]
            evec1, _ = eig_vecs[:, sort_indices]
            reduced_data = np.array(motion_data).dot(evec1)
            motion_estimation = reduced_data[-1]
            estimated_motion.append(motion_estimation)
        for i in features:
            (x, y) = i.ravel()
            cv2.circle(frame, (int(x), int(y)), 3, 255, (-1))
        cv2.imshow("Tracking", frame)
        k = cv2.waitKey(1000//30)
        if k == 27:
            break
    return motion_data, estimated_motion

def to_1d(motion):
    estimated_motion = deque()
    motion_data = deque()
    print(f"Motion length: {len(motion)}")
    while len(motion) != 0:
        motion_data.append(motion.popleft())
        if len(motion_data) >= 2:
            x, y = np.transpose(motion_data)
            coords = np.vstack([x, y])
            cov_mat = np.cov(coords)
            eig_vals, eig_vecs = np.linalg.eig(cov_mat)
            sort_indices = np.argsort(eig_vals)[::-1]
            evec1, _ = eig_vecs[:, sort_indices]
            reduced_data = np.array(motion_data).dot(evec1)
            motion_estimation = reduced_data[-1]
            estimated_motion.append(motion_estimation)
    return estimated_motion

def measureRR(motion_data, peak_minimum_sample_distance, t):
    freq = []
    peak_indices, fits = find_peaks(motion_data, peak_minimum_sample_distance, t)
    peak_times = np.take(t, peak_indices)
    diffs = [a - b for b, a in zip(peak_times, peak_times[1:])]
    print(f"Peak times: {peak_times}")
    if len(diffs) > 0:
        interval = np.mean(diffs)
        est_freq = 60.0 / interval  # 60 seconds in a minute
        freq.append(est_freq)
    return freq
    
def remove_outliers(data, m=2):
    data = data[abs(data - np.mean(data)) < m * np.std(data)]
    return data

if __name__ == "__main__":
    
    # cap = cv2.VideoCapture("../scp/case4.mp4")
    cap = cv2.VideoCapture(r"Segmented dataset\clip_0_(UAV_P03_1m_R_RGB)_30s.mp4")
    #cap = cv2.VideoCapture("/home/arnav/P1_Virtual_Dataset/extracted_cases/case22ugv_10sec.mp4")
    NUM_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    freq_max = 0.7
    timestamp = []
    

    print(f"Number of frames: {NUM_FRAMES}")
    # video = np.ndarray(shape=(NUM_FRAMES,int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),), dtype="float")
    body_vid = None
    chest_vid = None
    res_body = None
    res_chest = None
    frame_count = 0
    while cap.isOpened() and frame_count < NUM_FRAMES:
        ret, frame = cap.read()     
        if not ret:
            print("Not ret")
            break
        if len(timestamp) == 0:
            timestamp.append(0.)
        else:
            timestamp.append(timestamp[-1] + (1. / fps))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(f"Shape of gray frame: {gray.shape}")

        # cv2.imshow('Frame with ROIs', gray)
        if(frame_count==0):
            frameres=cv2.resize(frame,(800,800))
    
    
    # Break the loop if 'q' is pressed
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break
        # canny = cv2.Canny(gray, 100, 200)
        if res_chest is None:
            print(f"Frame size: {frame.shape}")
            chest_coords = chest(frame)
            body_coords = body(frame)
            res_chest = abs(chest_coords[0] - chest_coords[2]), abs(chest_coords[1] - chest_coords[3])
            res_body = abs(body_coords[0] - body_coords[2]), abs(body_coords[1] - body_coords[3])
            print(f"Res chest in first detect loop: {res_chest}")
            print(f"Res body in first detect loop: {res_body}")
        else:
            print(f"Res chest: {res_chest}")
            print(f"Res body: {res_body}")
            chest_coords = chest(frame, res_chest[0], res_chest[1])
            body_coords = body(frame, res_body[0], res_body[1])
        
        # print(f"Body coords: {body_coords}")
        # print(f"Chest coords: {chest_coords}")
        if(frame_count==0):
            frame_with_ROIs = draw_ROIs(frame, chest_coords, body_coords)
            
            # Display frame with ROIs
            cv2.imshow('Frame with ROIs', frame_with_ROIs)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
            
        if body_vid is None:
            body_vid = np.ndarray(shape=(NUM_FRAMES, body_coords[3] - body_coords[1], body_coords[2] - body_coords[0]), dtype="float")
            chest_vid = np.ndarray(shape=(NUM_FRAMES, chest_coords[3] - chest_coords[1], chest_coords[2] - chest_coords[0]), dtype="float")
            print(f"Shape of body_vid: {body_vid.shape}")
            print(f"Shape of chest_vid: {chest_vid.shape}")
        print(f"Chest coords before ROI: {chest_coords[1]}, {chest_coords[3]}")
        # chest_roi = gray[chest_coords[1]:chest_coords[3], chest_coords[0]:chest_coords[2]]

        chest_roi = gray[chest_coords[1]:chest_coords[3], chest_coords[0]:chest_coords[2]]
        print("Shape of chest roi: " + str(chest_roi.shape))
        chest_gray=gray[chest_coords[1]:chest_coords[3],chest_coords[0]:chest_coords[2]]
        gray_res=cv2.resize(chest_gray,res_chest)
        chest_vid[frame_count][:] = uint8_to_float(gray_res)
        body_vid[frame_count][:] = uint8_to_float(gray[body_coords[1]:body_coords[3],body_coords[0]:body_coords[2]])
        # cv2.imshow("Frame", gray)
        # cv2.imshow("Chest ROI", chest_roi)
        # cv2.imshow("Body ROI", gray[body_coords[1]:body_coords[3],body_coords[0]:body_coords[2]])
        # video[frame_count][:] = uint8_to_float(gray)
        frame_count += 1
        print(("Frame: " + str(frame_count)))

    print(f"Timestamp length: {len(timestamp)}")

    #Display chest video
    print("Displaying chest video")
    for frame in chest_vid:
        cv2.imshow('Chest Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# Display body_vid
    for frame in body_vid:
        cv2.imshow('Body Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    
    chest_motion_2d, chest_motion = track_motion(chest_vid, frame_count,pick_random=False)
    chest_motion_2d = np.array(chest_motion_2d)
    body_motion_2d, body_motion = track_motion(body_vid, frame_count,pick_random=True)
    body_motion_2d = np.array(body_motion_2d)
    
    print('Motion length: ' + str(len(chest_motion)))
    print(chest_motion)
    filtered_body_motion = butter_lowpass_filter(body_motion, 0.5, fps)
    filtered_chest_motion = butter_lowpass_filter(chest_motion, 0.5, fps)
    # Lowpass filtering chest
    lowpass_thresh = np.max(remove_outliers(filtered_chest_motion)) * 0.2
    print(f"Lowpass Threshold: {lowpass_thresh}")
    lowpass_chest_motion = np.where(filtered_chest_motion < lowpass_thresh, 0.0, filtered_chest_motion)
    x_motion = np.linspace(0, 100, len(chest_motion))
    
    body_motion_subtracted_2d = np.subtract(chest_motion_2d,body_motion_2d)
    print(body_motion_subtracted_2d.shape)
    #Convert body_motion_subtracted_2d to deque
    body_motion_subtracted_2d = deque([list(x) for x in body_motion_subtracted_2d])
    print(len(body_motion_subtracted_2d))
    
    body_motion_subtracted = to_1d(body_motion_subtracted_2d)
    
    body_motion_subtracted_filtered = butter_lowpass_filter(body_motion_subtracted, 0.5, fps)
    
    lowpass_subtracted_thresh = np.max(remove_outliers(body_motion_subtracted_filtered)) * 0.5
    print(f"Lowpass Subtracted Threshold: {lowpass_subtracted_thresh}")
    lowpass_subtracted_motion = np.where(body_motion_subtracted_filtered < lowpass_subtracted_thresh, 0.0, body_motion_subtracted_filtered)
    timestamp = timestamp[2:]
    print(f"Estimated RR: {measureRR(lowpass_subtracted_motion, int(np.floor(fps / freq_max)), timestamp)}")
    # plt.plEstimated RR: {amp, chest_raw, label="Raw")
    plt.plot(timestamp, filtered_chest_motion, label="Chest Motion Filtered")
    # plt.plot(timestamp, body_motion_subtracted, label="Subtracted Motion")
    plt.plot(timestamp, body_motion_subtracted_filtered, label="Subtracted Filtered")
    plt.plot(timestamp, filtered_body_motion, label="Body Motion Filtered")
    plt.plot(timestamp, lowpass_chest_motion, label="Chest Motion Filtered Lowpass")
    plt.plot(timestamp, lowpass_subtracted_motion, label="Subtracted Filtered Lowpass")
    # plt.plot(timestamp, body_motion, label="Body Motion")
    plt.legend()
    plt.show()