import cv2

image = cv2.imread('image processing/Assignment_26/6_death_symbol/image/MARYAM Mirzakhani.jpg')
image_2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite('image processing/Assignment_26/6_death_symbol/image/pic.jpg' , image_2)
rows = 675
cols = 1200

cv2.line(image_2 , (0,300) , (300,0) , 0 , 40)

cv2.imshow("death symbol",image_2)
cv2.waitKey()
cv2.imwrite('image processing/Assignment_26/6_death_symbol/output/MARYAM Mirzakhani.jpg',image_2)
