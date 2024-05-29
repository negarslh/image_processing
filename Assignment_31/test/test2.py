import cv2
import numpy as np
import matplotlib.pyplot as plt


image = np.array([[9,7,255,0],
                  [10,10,10,0],
                  [254,0,1,1],
                  [5,6,7,8]],dtype=np.uint8)

cv2.imwrite('output/image.png',image)

histogram = []

for i in range(256):
    histogram.append(np.sum(image == i))

plt.plot(histogram)
plt.show()