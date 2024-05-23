import cv2
import numpy as np

image = cv2.imread('3_Photo to Sketch/input/image.webp')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
inverted_image = 255 - gray_image
blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)
inverted_blurred = 255 - blurred
sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)

cv2.imwrite('3_Photo to Sketch/output/sketch_image.png', sketch)

cv2.imshow('Sketch Image', sketch)
cv2.waitKey(0)
cv2.destroyAllWindows()
