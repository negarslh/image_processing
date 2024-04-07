import cv2
import numpy as np

width = 500
height = 500

gradient_img = np.zeros((height, width), dtype=np.uint8)

for y in range(height):
    for x in range(width):
        gradient_img[y, x] = int(255 * (height - y) / height)  

cv2.imshow('Gradient', gradient_img)
cv2.waitKey(0)
cv2.imwrite('image processing/Assignment_26/5_generate_a_gradient/output/gradient.jpg', gradient_img)
