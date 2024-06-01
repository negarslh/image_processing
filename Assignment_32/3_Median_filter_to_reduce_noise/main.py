import cv2
import numpy as np
import os

def apply_filters(image, kernel_size):
    list_of_filters = []

    median = cv2.medianBlur(image, kernel_size)

    list_of_filters.extend([image, median])

    return list_of_filters

image_paths = [
    "3_Median_filter_to_reduce_noise/input/noisy_board.png",
    "3_Median_filter_to_reduce_noise/input/noisy_family.png",
    "3_Median_filter_to_reduce_noise/input/noisy_girl.png",
    "3_Median_filter_to_reduce_noise/input/noisy_image.png",
    "3_Median_filter_to_reduce_noise/input/noisy_picture.png",
    "3_Median_filter_to_reduce_noise/input/noisy_skeleton.png"
]

kernel_size = 5

for i, image_path in enumerate(image_paths):
    if not os.path.exists(image_path):
        print(f"File {image_path} does not exist.")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image {image_path}")
        continue

    filters = apply_filters(image, kernel_size)

    combined_filters = np.hstack(filters)

    output_path = f"3_Median_filter_to_reduce_noise/output/combined_filters_{i + 1}.jpg"
    cv2.imwrite(output_path, combined_filters)

    cv2.imshow(f"Combined Filters {i + 1}", combined_filters)
    cv2.waitKey(0)

cv2.destroyAllWindows()
