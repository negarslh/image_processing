import cv2
import numpy as np

image_1 = cv2.imread('4_Find the secret text/input/image_1.png')
image_2 = cv2.imread('4_Find the secret text/input/image_2.png')

image_1 = image_1.astype(np.float32)
image_2 = image_2.astype(np.float32)

result = cv2.subtract(image_1, image_2)
result = result.astype(np.uint8)

cv2.imwrite('4_Find the secret text/output/result.png', result)

cv2.imshow('Result', result)
cv2.waitKey()
