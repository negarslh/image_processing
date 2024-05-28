import cv2
import numpy as np

from TFLiteFaceDetector import UltraLightFaceDetecion
from TFLiteFaceAlignment import CoordinateAlignmentModel

fd = UltraLightFaceDetecion("1_Snapchat_Filter/weights/RFB-320.tflite", conf_threshold=0.88)
fa = CoordinateAlignmentModel("1_Snapchat_Filter/weights/coor_2d106.tflite")

image = cv2.imread('1_Snapchat_Filter/input/face.jpg')
apple = cv2.imread('1_Snapchat_Filter/input/apple.jpg')
orange = cv2.imread('1_Snapchat_Filter/input/orange.jpg')

boxes, scores = fd.inference(image)
for pred in fa.get_landmarks(image, boxes):

    lips_landmarks = np.array([pred[i] for i in [52, 55, 56, 53, 59, 58, 61, 68, 67, 71, 63, 64]], dtype=int)
    right_eye_landmarks = np.array([pred[i] for i in [39, 42, 40, 41, 35, 36, 33, 37]], dtype=int)
    left_eye_landmarks = np.array([pred[i] for i in [89, 90, 87, 91, 93, 96, 94, 95]], dtype=int)
    
mask = np.zeros(image.shape , dtype=np.uint8)
cv2.drawContours(mask, [lips_landmarks , right_eye_landmarks , left_eye_landmarks], -1, (255, 255, 255), -1)

mask = mask / 255
lip_eyes = image * mask

lip_eyes_resized_apple = cv2.resize(lip_eyes, (apple.shape[1], apple.shape[0]))
lip_eyes_resized_orange = cv2.resize(lip_eyes, (orange.shape[1], orange.shape[0]))

apple_filter = apple.copy()
orange_filter = orange.copy()

resized_mask_apple = cv2.resize(mask, (lip_eyes_resized_apple.shape[1], lip_eyes_resized_apple.shape[0]))
apple_filter[resized_mask_apple.astype(bool)] = lip_eyes_resized_apple[resized_mask_apple.astype(bool)]

resized_mask_orange = cv2.resize(mask, (lip_eyes_resized_orange.shape[1], lip_eyes_resized_orange.shape[0]))
orange_filter[resized_mask_orange.astype(bool)] = lip_eyes_resized_orange[resized_mask_orange.astype(bool)]

cv2.imwrite('1_Snapchat_Filter/output/apple_filter.png',apple_filter)
cv2.imwrite('1_Snapchat_Filter/output/orange_filter.png',orange_filter)

cv2.imshow("apple_filter", apple_filter)
cv2.imshow("orange_filter", orange_filter)
cv2.waitKey(0)
cv2.destroyAllWindows()
