import cv2
import numpy as np

image = cv2.imread('input/airplane.jpg', cv2.IMREAD_GRAYSCALE)
rows, cols = image.shape
result = np.zeros((rows, cols),dtype=np.uint8)

# filter = np.array([[1/9, 1/9, 1/9],
#                    [1/9, 1/9, 1/9],
#                    [1/9, 1/9, 1/9]])

filter = np.ones((5,5))/25

for i in range(2, rows-2):
    for j in range(2, cols-2):
        small = image[i-2:i+3, j-2:j+3] #crop kardan ye nahie kochik az image
        # average = np.mean(small)
        average = np.sum(small * filter)
        result[i, j] = int(average)

cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite('output/airplane.jpg', result)
cv2.imwrite('output/airplane5.jpg', result)
