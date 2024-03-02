import cv2
import numpy as np


img = cv2.imread('../datasets/facial_color/001.jpg')
K = 4

np.random.seed(42)
colors = np.random.randint(0,256,(K,3))
colors = [[255,0,0], [0,255,0], [0,0,255], [128,128,128]]
print(colors)

pixel_vals = img.reshape((-1,3))
pixel_vals = np.float32(pixel_vals)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
retval, labels, centers = cv2.kmeans(pixel_vals, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((img.shape))
print(np.unique(segmented_data, axis=0).shape)
for (i,val) in enumerate(np.unique(segmented_data, axis=0)):
    segmented_image[np.all(segmented_image == val, axis=-1)] = colors[i]
    cv2.waitKey(0)
print(segmented_image.shape)
cv2.imshow("Original Image",img)
cv2.imshow('segmented_image', segmented_image)
cv2.waitKey(0)
