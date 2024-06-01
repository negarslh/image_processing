import cv2
import numpy as np
import os

def equalizer_histogram(image, name):
    equalized = cv2.equalizeHist(image)

    equal_hist_result = np.hstack((image, equalized))

    cv2.imwrite(
        f"4_Histogram_Equalization/output/equal_hist_{name}.png",
        equal_hist_result,
    )
    return equal_hist_result

def clahe(image , name):
    list_of_filters = []

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_applied = clahe.apply(image)

    clahe_result = np.hstack((image, clahe_applied))

    cv2.imwrite(
        f"4_Histogram_Equalization/output/clahe_{name}.png",
        clahe_result,
    )

    list_of_filters.extend([image, clahe_result])

    return list_of_filters

image_paths = [
    "4_Histogram_Equalization/input/city.png",
    "4_Histogram_Equalization/input/image.png",
    "4_Histogram_Equalization/input/plain.png"
]

for i, image_path in enumerate(image_paths):
    if not os.path.exists(image_path):
        print(f"File {image_path} does not exist.")
        continue

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to read image {image_path}")
        continue

    name = os.path.splitext(os.path.basename(image_path))[0]

    equal_hist_filters = equalizer_histogram(image, name)
    combined_equal_hist_filters = np.hstack(equal_hist_filters)

    clahe_filters = clahe(image, name)
    combined_clahe_filters = np.hstack(clahe_filters)

cv2.destroyAllWindows()
