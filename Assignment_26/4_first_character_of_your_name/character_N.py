import cv2
import numpy as np

image = np.ones((500, 500), np.uint8) * 255

cv2.line(image , (150 , 400) , (150 , 100) , 128 , 10)
cv2.line(image , (150 , 100) , (350 , 400) , 128 , 10)
cv2.line(image , (350 , 400) , (350 , 100) , 128 , 10)

cv2.imshow('character', image)
cv2.imwrite('image processing/Assignment_26/4_first_character_of_your_name/output/character.jpg', image)
cv2.waitKey()