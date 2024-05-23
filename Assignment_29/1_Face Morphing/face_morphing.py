import cv2
import numpy as np

image_1 = cv2.imread('1_Face Morphing/input/img1.jpg')
image_2 = cv2.imread('1_Face Morphing/input/img2.jpg')

image_1 = image_1.astype(np.float32)
image_2 = image_2.astype(np.float32)

morph_25 = (image_1 * 0.25) + (image_2 * 0.75)
morph_50 = (image_1 * 0.5) + (image_2 * 0.5)
morph_75 = (image_1 * 0.75) + (image_2 * 0.25)

morph_25 = morph_25.astype(np.uint8)
morph_50 = morph_50.astype(np.uint8)
morph_75 = morph_75.astype(np.uint8)

combined_image_sequence = np.concatenate(
    (image_1, morph_75, morph_50, morph_25, image_2),
    axis=1
)

cv2.imwrite("1_Face Morphing/output/combined_image.jpeg", combined_image_sequence)
