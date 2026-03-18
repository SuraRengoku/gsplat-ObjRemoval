import cv2
import numpy as np

img = np.zeros((100, 100), dtype=np.uint8)
img[40:60, 40:60] = 1 # ipmask

fg = np.zeros((100, 100), dtype=np.uint8)
fg[35:65, 35:65] = 1 # fgmask larger

# First expand a bit
kernel = np.ones((5,5), np.uint8)

curr_mask = img.copy()

for i in range(10):
    prev_sum = curr_mask.sum()
    dilated = cv2.dilate(curr_mask, kernel, iterations=1)
    curr_mask = cv2.bitwise_and(dilated, fg)
    if curr_mask.sum() == prev_sum: # Converged
        break

print("Converged at iteration:", i)

