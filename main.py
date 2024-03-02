from transforms import eulerian_magnification_bandpass
import cv2
import numpy as np
from transforms import float_to_uint8, uint8_to_float

def trackMotion(ROI):
    features = cv2.goodFeaturesToTrack(ROI, 100, 0.01, 10)

cap = cv2.VideoCapture('../scp/ugv1.mp4')

CALL_FRAMES = 300

video = np.ndarray(shape=(CALL_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))), dtype="float")
frame_count = 0

while cap.isOpened() and frame_count < CALL_FRAMES:
    ret, frame = cap.read()
    cv2.waitKey(1)
    if ret:
        global gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        video[frame_count][:] = uint8_to_float(gray)
        
        frame_count += 1
        print("Frame: " + str(frame_count))
    else:
        break

print(video.shape)
evm, raw = eulerian_magnification_bandpass(video, fps=10, freq_min=0.1, freq_max=1, amplification=500, pyramid_levels=9, skip_levels_at_top=4, threshold=0.7)
avg_frame = np.array(np.average(evm, axis=0))
avg_norm = ((avg_frame - avg_frame.min()) / (avg_frame.max() - avg_frame.min()))
avg = float_to_uint8(avg_norm)
ret, thresh = cv2.threshold(avg, 20, 255, cv2.THRESH_BINARY)

cv2.imshow("Frame", gray)
cv2.imshow("Average EVM", avg)
cv2.imshow("Threshold", thresh)
# cv2.imshow("EVM", evm)
cv2.waitKey(0)
cv2.destroyAllWindows()

