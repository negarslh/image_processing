import cv2
import numpy as np

def detect_edges(image):

    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError("Unable to read the image. Check the path and try again.")
    
    rows , cols = image.shape

    result = np.zeros((rows, cols),dtype=np.uint8)

    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    ### The first method
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            small = image[i-1:i+2, j-1:j+2] 
            result[i, j] = np.abs(np.sum(small * kernel))
    
    ### The second method
    result = cv2.filter2D(image, -1, kernel)
    
    return result

image = '3_Edge detection/input/image.png'

edges_image = detect_edges(image)

cv2.imwrite('3_Edge detection/output/edges.png', edges_image)
cv2.imshow('Edges', edges_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
