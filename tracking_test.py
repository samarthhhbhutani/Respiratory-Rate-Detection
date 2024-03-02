from collections import deque

import cv2
import numpy as np
from matplotlib import pyplot as plt
from transforms import butter_lowpass_filter, float_to_uint8, uint8_to_float
import peakutils

def find_peaks(filtered_data, peak_minimum_sample_distance, t):
    gaussian_cutoff = 10.0
    width = peak_minimum_sample_distance
    print(width)
    indices = peakutils.indexes(filtered_data, min_dist=width)

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



def track_motion(ROI, pick_random=False):
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
            if not pick_random:
                features = cv2.goodFeaturesToTrack(frame, 20, 0.01, 4, blockSize=3)
            else:
                #Picking 10 random points as features
                features = np.random.randint(0, 50, size=(10, 1, 2)).astype(np.float32)
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
        k = cv2.waitKey(0)
        if k == 27:
            break
    return motion_data, estimated_motion

def measureRR(motion_data, peak_minimum_sample_distance, t):
    freq = []
    peak_indices, fits = find_peaks(motion_data, peak_minimum_sample_distance, t)
    peak_times = np.take(t, peak_indices)
    diffs = [a - b for b, a in zip(peak_times, peak_times[1:])]
    if len(diffs) > 0:
        interval = np.mean(diffs)
        est_freq = 60.0 / interval  # 60 seconds in a minute
        freq.append(est_freq)
    return freq
    



if __name__ == "__main__":
    cap = cv2.VideoCapture("../scp/ugv1_small_30sec.mp4")
    # cap = cv2.VideoCapture("./case4_30sec.mp4")
    
    NUM_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    freq_max = 0.7
    timestamp = []
    print(f"Number of frames: {NUM_FRAMES}")
    # video = np.ndarray(shape=(NUM_FRAMES,int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),), dtype="float")
    body_vid = np.ndarray(shape=(NUM_FRAMES, 230, 250), dtype="float")
    chest_vid = np.ndarray(shape=(NUM_FRAMES, 50, 50), dtype="float")
    frame_count = 0
    while cap.isOpened() and frame_count < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if len(timestamp) == 0:
            timestamp.append(0.)
        else:
            timestamp.append(timestamp[-1] + (1. / fps))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # canny = cv2.Canny(gray, 100, 200)
        chest_vid[frame_count][:] = uint8_to_float(gray[300:350,265:315])
        body_vid[frame_count][:] = uint8_to_float(gray[250:480, 200:450])
        cv2.imshow("Frame", gray)
        # video[frame_count][:] = uint8_to_float(gray)
        frame_count += 1
        print(("Frame: " + str(frame_count)))

    print(f"Timestamp length: {len(timestamp)}")
    _, chest_motion = track_motion(chest_vid)
    _, body_motion = track_motion(body_vid)
    print('Motion length: ' + str(len(chest_motion)))
    print(chest_motion)
    filtered_body_motion = butter_lowpass_filter(body_motion, 0.5, fps)
    filtered_chest_motion = butter_lowpass_filter(chest_motion, 0.5, fps)
    # Lowpass filtering chest
    lowpass_thresh = np.max(filtered_chest_motion[abs(filtered_chest_motion - np.mean(filtered_chest_motion)) < 2 * np.std(filtered_chest_motion)]) * 0.4
    print(f"Lowpass Threshold: {lowpass_thresh}")
    lowpass_chest_motion = np.where(filtered_chest_motion < lowpass_thresh, 0.0, filtered_chest_motion)
    
    x_motion = np.linspace(0, 100, len(chest_motion))
    
    body_motion_subtracted = np.subtract(chest_motion,body_motion)
    body_motion_subtracted_filtered = butter_lowpass_filter(body_motion_subtracted, 0.5, fps)
    
    timestamp = timestamp[2:]
    print(f"Estimated RR: {measureRR(lowpass_chest_motion, int(np.floor(fps / freq_max)), timestamp)}")
    # plt.plEstimated RR: {amp, chest_raw, label="Raw")
    plt.plot(timestamp, filtered_chest_motion, label="Chest Motion Filtered")
    # plt.plot(timestamp, body_motion_subtracted, label="Subtracted Motion")
    plt.plot(timestamp, body_motion_subtracted_filtered, label="Subtracted Filtered")
    plt.plot(timestamp, filtered_body_motion, label="Body Motion Filtered")
    plt.plot(timestamp, lowpass_chest_motion, label="Chest Motion Filtered Lowpass")
    # plt.plot(timestamp, body_motion, label="Body Motion")
    plt.legend()
    plt.show()