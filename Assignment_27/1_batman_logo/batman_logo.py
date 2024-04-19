from PIL import Image
import pillow_avif
import cv2

#convert to jpg format
image = Image.open('image processing/Assignment_27/1_batman_logo/image/image.avif')
image.save('image processing/Assignment_27/1_batman_logo/image/image.jpg')

image = cv2.imread('image processing/Assignment_27/1_batman_logo/image/image.jpg')

#convert to gray
gray_logo = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('image processing/Assignment_27/1_batman_logo/output/gray_logo.jpg' , gray_logo)


threshold = 100
cols , rows = gray_logo.shape

_ , logo = cv2.threshold(gray_logo , threshold , 255 , cv2.THRESH_BINARY_INV)

cv2.putText(logo, "BATMAN", (rows - 200, cols - 20 ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255 , 255 , 255))

cv2.imshow('logo', logo)
cv2.waitKey()

cv2.imwrite('image processing/Assignment_27/1_batman_logo/output/logo.jpg' , logo)

