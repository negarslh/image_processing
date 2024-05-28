import numpy as np
import cv2
import tensorflow as tf
import time

from TFLiteFaceDetector import UltraLightFaceDetecion
from TFLiteFaceAlignment import CoordinateAlignmentModel

def enlarge_feature(image, landmarks, scale):
    x, y, w, h = cv2.boundingRect(landmarks)

    mask = np.zeros_like(image)
    cv2.drawContours(mask, [landmarks], -1, (1, 1, 1), -1)
    
    feature = cv2.resize(image[y:y+h, x:x+w] * mask[y:y+h, x:x+w], (0, 0), fx=scale, fy=scale)

    bh, bw = feature.shape[:2]
    cx, cy = x + w // 2, y + h // 2
    nx, ny = max(cx - bw // 2, 0), max(cy - bh // 2, 0)

    if nx + bw > image.shape[1]: bw = image.shape[1] - nx
    if ny + bh > image.shape[0]: bh = image.shape[0] - ny

    mask_resized = cv2.resize(mask[y:y+h, x:x+w], (bw, bh))

    image[ny:ny+bh, nx:nx+bw] = (
        image[ny:ny+bh, nx:nx+bw] * (1 - mask_resized) +
        feature[:bh, :bw] * mask_resized
    )

    return image


fd = UltraLightFaceDetecion("4_Big_Eyes_And_Lips/weights/RFB-320.tflite", conf_threshold=0.88)
fa = CoordinateAlignmentModel("4_Big_Eyes_And_Lips/weights/coor_2d106.tflite")

image = cv2.imread('4_Big_Eyes_And_Lips/input/image.jpg')

start_time = time.perf_counter()

boxes, scores = fd.inference(image)

for pred in fa.get_landmarks(image, boxes):

    lips_landmarks = np.array([pred[i] for i in [52, 55, 56, 53, 59, 58, 61, 68, 67, 71, 63, 64]], dtype=int)
    image = enlarge_feature(image, lips_landmarks, scale=2)

    right_eye_landmarks = np.array([pred[i] for i in [39, 42, 40, 41, 35, 36, 33, 37]], dtype=int)
    image = enlarge_feature(image, right_eye_landmarks, scale=2)

    left_eye_landmarks = np.array([pred[i] for i in [89, 90, 87, 91, 93, 96, 94, 95]], dtype=int)
    image = enlarge_feature(image, left_eye_landmarks, scale=2)

print(f"Processing time: {time.perf_counter() - start_time:.2f} seconds")

cv2.imwrite('4_Big_Eyes_And_Lips/output/image.png', image)
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
