import cv2
import numpy as np

from TFLiteFaceDetector import UltraLightFaceDetecion
from TFLiteFaceAlignment import CoordinateAlignmentModel

fd = UltraLightFaceDetecion("2_Rotate_Photo/weights/RFB-320.tflite", conf_threshold=0.88)
fa = CoordinateAlignmentModel("2_Rotate_Photo/weights/coor_2d106.tflite")

def rotate_photo(image, landmarks):
    x, y, w, h = cv2.boundingRect(landmarks)

    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [landmarks], -1, (255, 255, 255), -1)
    mask = mask // 255

    result = image * mask
    result = result[y:y+h, x:x+w]

    rotated_feature = cv2.rotate(result,cv2.ROTATE_180)

    rotated_feature_height, rotated_feature_width = rotated_feature.shape[:2]
    center_x, center_y = x + w // 2, y + h // 2
    new_x = max(center_x - rotated_feature_width // 2, 0)
    new_y = max(center_y - rotated_feature_height // 2, 0)

    max_width = image.shape[1] - new_x
    max_height = image.shape[0] - new_y
    rotated_feature_width = min(rotated_feature_width, max_width)
    rotated_feature_height = min(rotated_feature_height, max_height)

    rotated_feature_mask = cv2.resize(mask[y:y+h, x:x+w], (rotated_feature_width, rotated_feature_height))

    image[new_y:new_y+rotated_feature_height, new_x:new_x+rotated_feature_width] = (
        image[new_y:new_y+rotated_feature_height, new_x:new_x+rotated_feature_width] * (1 - rotated_feature_mask) +
        rotated_feature[:rotated_feature_height, :rotated_feature_width] * rotated_feature_mask
    )


    image = cv2.rotate(image , cv2.ROTATE_180)
    return image

image = cv2.imread('2_Rotate_Photo/input/image.jpg')

boxes, scores = fd.inference(image)

for pred in fa.get_landmarks(image, boxes):

    lips_landmarks = np.array([pred[i] for i in [52, 55, 56, 53, 59, 58, 61, 68, 67, 71, 63, 64]], dtype=int)
    rotate_image = rotate_photo(image, lips_landmarks)

    right_eye_landmarks = np.array([pred[i] for i in [39, 42, 40, 41, 35, 36, 33, 37]], dtype=int)
    rotate_image = rotate_photo(image, right_eye_landmarks)

    left_eye_landmarks = np.array([pred[i] for i in [89, 90, 87, 91, 93, 96, 94, 95]], dtype=int)
    rotate_image = rotate_photo(image, left_eye_landmarks)  
    
cv2.imwrite('2_Rotate_Photo/output/rotate_image.jpg', rotate_image)  
cv2.imshow("Rotated Image", rotate_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
