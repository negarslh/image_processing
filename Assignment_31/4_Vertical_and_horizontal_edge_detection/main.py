import cv2
import numpy as np

def detect_vertical_horizontal_edges(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError("Unable to read the image. Check the path and try again.")
    
    rows, cols = image.shape
    vertical_result = np.zeros((rows, cols), dtype=np.uint8)
    horizontal_result = np.zeros((rows, cols), dtype=np.uint8)

    kernel_vertical = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])
    
    kernel_horizontal = np.array([[ 1,  2,  1],
                                  [ 0,  0,  0],
                                  [-1, -2, -1]])
    
    ### The first method
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            small = image[i-1:i+2, j-1:j+2]
            vertical_result[i, j] = np.clip(np.abs(np.sum(small * kernel_vertical)), 0, 255)
    
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            small = image[i-1:i+2, j-1:j+2]
            horizontal_result[i, j] = np.clip(np.abs(np.sum(small * kernel_horizontal)), 0, 255)
    
    ### The second method
    vertical_result = cv2.filter2D(image, -1, kernel_vertical)
    horizontal_result = cv2.filter2D(image, -1, kernel_horizontal)

    edges_combined = np.clip((vertical_result / 2 + horizontal_result / 2), 0, 255).astype(np.uint8)
    
    return vertical_result, horizontal_result, edges_combined

image_path = '4_Vertical and horizontal edge detection/input/image.jpg'
edges_vertical, edges_horizontal, edges_combined = detect_vertical_horizontal_edges(image_path)

cv2.imwrite('4_Vertical and horizontal edge detection/output/vertical_edges.png', edges_vertical)
cv2.imwrite('4_Vertical and horizontal edge detection/output/horizontal_edges.png', edges_horizontal)
cv2.imwrite('4_Vertical and horizontal edge detection/output/combined_edges.png', edges_combined)

cv2.imshow('Vertical Edges', edges_vertical)
cv2.imshow('Horizontal Edges', edges_horizontal)
cv2.imshow('Combined Edges', edges_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
