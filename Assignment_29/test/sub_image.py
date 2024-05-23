import cv2
import numpy as np

image_1 = cv2.imread('test/input/dream1.png')
image_2 = cv2.imread('test/input/dream2.png')

image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)


result = image_1 - image_2

cv2.imshow('result', result)
cv2.waitKey()