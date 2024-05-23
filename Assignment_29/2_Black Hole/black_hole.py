import os
import cv2
import numpy as np
import random

folder_path_1 = '2_Black Hole/input/black_hole/1'
folder_path_2 = '2_Black Hole/input/black_hole/2'
folder_path_3 = '2_Black Hole/input/black_hole/3'
folder_path_4 = '2_Black Hole/input/black_hole/4'

image_1 = []
image_2 = []
image_3 = []
image_4 = []

for image_path in [folder_path_1 , folder_path_2 , folder_path_3 , folder_path_4] :
    image = cv2.imread("2_Black Hole/input/black_hole/"+image_path).astype(np.float32)

    image_1.append(image)
    image_2.append(image)
    image_3.append(image)
    image_4.append(image)

result = np.zeros(image.shape)

for image in image_1 :
    result += image

for image in image_2 :
    result += image

for image in image_3 :
    result += image

for image in image_4 :
    result += image

    