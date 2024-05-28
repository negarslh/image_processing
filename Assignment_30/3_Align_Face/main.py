import cv2
import numpy as np

from TFLiteFaceDetector import UltraLightFaceDetecion
from TFLiteFaceAlignment import CoordinateAlignmentModel

fd = UltraLightFaceDetecion("2_Rotate_Photo/weights/RFB-320.tflite", conf_threshold=0.88)
fa = CoordinateAlignmentModel("2_Rotate_Photo/weights/coor_2d106.tflite")

def align_face(image, left_eye_landmarks, right_eye_landmarks, lips_landmarks):
    left_eye_center = np.mean(left_eye_landmarks, axis=0).astype(int)
    right_eye_center = np.mean(right_eye_landmarks, axis=0).astype(int)

    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]

    angle = np.degrees(np.arctan2(dy, dx)) - 180

    eyes_center = (int(right_eye_center[0]), int(right_eye_center[1]))

    rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale=1)

    aligned_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

    return aligned_image

image = cv2.imread('3_Align_Face/input/image.jpg')

boxes, scores = fd.inference(image)

for pred in fa.get_landmarks(image, boxes):
    lips_landmarks = np.array([pred[i] for i in [52, 55, 56, 53, 59, 58, 61, 68, 67, 71, 63, 64]], dtype=int)
    right_eye_landmarks = np.array([pred[i] for i in [39, 42, 40, 41, 35, 36, 33, 37]], dtype=int)
    left_eye_landmarks = np.array([pred[i] for i in [89, 90, 87, 91, 93, 96, 94, 95]], dtype=int)

    image = align_face(image, left_eye_landmarks, right_eye_landmarks, lips_landmarks)

cv2.imwrite('3_Align_Face/output/face.jpg', image)
cv2.imshow("aligned Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
