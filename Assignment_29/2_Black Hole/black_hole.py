import os
import numpy as np
from PIL import Image
import cv2

def calculate_mean_image(directory):
    image_arrays = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            image_arrays.append(image_array)
    
    if not image_arrays:
        raise ValueError("No images found in the directory.")
    
    mean_image_array = np.mean(image_arrays, axis=0).astype(np.uint8)
    
    mean_image = Image.fromarray(mean_image_array)
    
    return mean_image

def process_multiple_directories(base_directory, subdirectories):
    all_mean_images_cv2 = []
    
    for subdirectory in subdirectories:
        directory_path = os.path.join(base_directory, subdirectory)
        try:
            mean_image = calculate_mean_image(directory_path)
            mean_image_cv2 = cv2.cvtColor(np.array(mean_image), cv2.COLOR_RGB2BGR)
            cv2.waitKey(0) 
            cv2.destroyAllWindows()
            
            all_mean_images_cv2.append(mean_image_cv2)
        except ValueError as e:
            print(f"Skipping {subdirectory}: {e}")
    
    return all_mean_images_cv2

base_directory = '2_Black Hole/input/black_hole'
subdirectories = ['1', '2', '3', '4']

mean_images_list = process_multiple_directories(base_directory, subdirectories)

combined_image_up = np.concatenate(mean_images_list[:2], axis=1)
combined_image_down = np.concatenate(mean_images_list[2:], axis=1)

black_hole = np.concatenate((combined_image_up, combined_image_down), axis=0)

cv2.imwrite('2_Black Hole/output/black_hole.jpg', black_hole)

cv2.imshow("black_hole", black_hole)
cv2.waitKey(0)
cv2.destroyAllWindows()
