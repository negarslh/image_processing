import cv2

image = cv2.imread('image processing/Assignment_26/3_rotate_image/images/3.jpg')

if image is not None :
    rotated_image = cv2.rotate(image, cv2.ROTATE_180)

    cv2.imshow('Original Image', image)
    cv2.imshow('Rotated Image', rotated_image)
    cv2.waitKey()
    cv2.imwrite('image processing/Assignment_26/3_rotate_image/output/rotated_image.jpg', rotated_image)

else:
    print('Image is not found!')