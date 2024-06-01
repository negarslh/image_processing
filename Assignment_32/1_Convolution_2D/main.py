import cv2
import numpy as np

def filtering(input_image):
    kernels = [
        # 1. Edge detection filter
        np.array([[-1, -1, -1],
                  [-1,  8, -1],
                  [-1, -1, -1]]),

        # 2. Sharpening filter
        np.array([[0, -1, 0],
                  [-1, 5, -1],
                  [0, -1, 0]]),

        # 3. Emboss filter
        np.array([[-2, -1, 0],
                  [-1, 1, 1],
                  [0, 1, 2]]),

        # 4. Identity filter
        np.array([[0, 0, 0],
                  [0, 1, 0],
                  [0, 0, 0]]),

        # 5. Custom filter
        np.array([[1, 0, 1],
                  [0, 0, 0],
                  [1, 0, 1]])
    ]

    for i, kernel in enumerate(kernels):
        output_image = cv2.filter2D(input_image, -1, kernel)
        combined_result = np.hstack((input_image, output_image))
        cv2.imwrite(f'1_Convolution_2D/output/result_{i+1}.jpg', combined_result)
        cv2.imshow(f'result_{i+1}', combined_result)
        cv2.waitKey(0)

# Load the input image
input_image = cv2.imread('1_Convolution_2D/input/image.jpg')

if input_image is not None:
    # Apply the filtering function
    filtering(input_image)
    print("Processing completed and results saved.")
else:
    print("Error: Input image not found or could not be read.")
