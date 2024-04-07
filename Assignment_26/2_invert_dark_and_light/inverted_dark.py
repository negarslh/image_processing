import cv2

image = cv2.imread('image processing/Assignment_26/2_invert_dark_and_light/images/1.jpg')

if image is not None and image.size != 0:
    inverted_image = 255 - image

    cv2.imshow('Original Image', image)
    cv2.imshow('Inverted Image', inverted_image)
    cv2.waitKey()
    cv2.imwrite('image processing/Assignment_26/2_invert_dark_and_light/output/inverted_dark.jpg', inverted_image)

else:
    print('Image not found!')