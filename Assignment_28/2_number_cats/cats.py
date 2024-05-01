import cv2

image = cv2.imread('2_number_cats/input/cats.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')
faces = face_detector.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=5)

for face in faces:
    x, y, w, h = face
    cv2.rectangle(image, [x, y] , [x+w, y+h] , (0, 255, 0), 2)

    
print("Number of cats Face : " , len(faces))
cv2.imshow('Cat Face Detector', image)
cv2.waitKey()
cv2.destroyAllWindows()
