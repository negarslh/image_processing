import cv2
import numpy as np

def mean_filter(image_path, kernel_size):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Unable to read the image at {image_path}. Check the path and try again.")
    
    filtered_image = cv2.blur(image, (kernel_size, kernel_size))
    
    return filtered_image

image_paths = [
    '5_Noise reduction/input/noisy_board.png',
    '5_Noise reduction/input/noisy_image.png',
    '5_Noise reduction/input/noisy_skeleton.png'
]

kernel_sizes = [3, 5, 15]

for image_path in image_paths:
    for kernel_size in kernel_sizes:
        
        filtered_image = mean_filter(image_path, kernel_size)
        
        output_path = f'5_Noise reduction/output/{image_path.split("/")[-1].split(".")[0]}_filtered_{kernel_size}x{kernel_size}.png'
        cv2.imwrite(output_path, filtered_image)
        
        cv2.imshow(f'Filtered Image {kernel_size}x{kernel_size}', filtered_image)
        cv2.waitKey(0)

cv2.destroyAllWindows()
