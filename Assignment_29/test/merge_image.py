import cv2
import numpy as np

ana = cv2.imread('test/input/ana_face.png')
lion = cv2.imread('test/input/lion.png')

ana = cv2.cvtColor(ana, cv2.COLOR_BGR2GRAY)
lion = cv2.cvtColor(lion, cv2.COLOR_BGR2GRAY)

ana = ana.astype(np.float32)
lion = lion.astype(np.float32)

result = ana*2/5 + lion*4/5

result = result.astype(np.uint8)

cv2.imshow('result', result)
cv2.waitKey()
