import cv2
import numpy as np

image_1 = cv2.imread('test/input/w1.jpg')
image_2 = cv2.imread('test/input/w2.jpg')

image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

# result = cv2.add(image_1, image_2)
# result = np.add(image_1, image_2)

image_1 = image_1.astype(np.float32)
image_2 = image_2.astype(np.float32)

result = image_1 + image_2
result = result.astype(np.uint8)

cv2.imwrite('test/output/add_output.jpg' , result)

cv2.imshow('result', result)
cv2.waitKey()

