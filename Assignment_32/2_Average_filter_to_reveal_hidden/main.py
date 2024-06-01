import cv2
import numpy as np

def apply_filters(image, kernel_size):
    list_of_filters = []

    kernel_normal = np.ones((kernel_size, kernel_size), dtype=np.float32) / 5
    kernel_high = np.ones((kernel_size, kernel_size), dtype=np.float32) / 1
    kernel_low = np.ones((kernel_size, kernel_size), dtype=np.float32) / 0.04

    result_normal = cv2.filter2D(image, -1, kernel_normal)
    result_high = cv2.filter2D(image, -1, kernel_high)
    result_low = cv2.filter2D(image, -1, kernel_low)

    list_of_filters.extend([image, result_normal, result_high, result_low])

    return list_of_filters

image = cv2.imread("2_Average_filter_to_reveal_hidden/input/image.tif")

filters_3x3 = apply_filters(image, 3)
filters_5x5 = apply_filters(image, 5)

combined_filters_3x3 = np.hstack(filters_3x3)
combined_filters_5x5 = np.hstack(filters_5x5)
combined_all = np.vstack((combined_filters_3x3, combined_filters_5x5))

cv2.imwrite("2_Average_filter_to_reveal_hidden/output/image.tif", combined_all)

cv2.imshow("Combined Filters", combined_all)
cv2.waitKey(0)
cv2.destroyAllWindows()
