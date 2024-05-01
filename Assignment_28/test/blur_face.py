import cv2

image = cv2.imread('test/angelina.jpg')

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image_gray = cv2.resize(image_gray, (400, 400))
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces = face_detector.detectMultiScale(image_gray)

for face in faces:
    x, y, w, h = face
    # cv2.rectangle(image_gray , (x, y), (x+w, y+h), 0 , 5)

    face_image = image_gray[y:y+h , x:x+w]
    face_image_small = cv2.resize(face_image, [20,20])
    face_image_big = cv2.resize(face_image_small, [w,h] , interpolation=cv2.INTER_NEAREST)
    image_gray[y:y+h , x:x+w] = face_image_big

print(faces)
# print(image_gray[y:y+h , x:x+w])

cv2.imshow('angelina' , image_gray)
cv2.waitKey()
