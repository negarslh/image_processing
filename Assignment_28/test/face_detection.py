import cv2

image = cv2.imread('test/Angelina-Jolie.webp')
img_sticker = cv2.imread('test/sticker.png')

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray = cv2.resize(image_gray, (400, 400))
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces = face_detector.detectMultiScale(image_gray)

for face in faces:
    x, y, w, h = face
    # cv2.rectangle(image_gray , (x, y), (x+w, y+h), 0 , 5)
    sticker = cv2.resize(img_sticker,[w, h])
    image_gray[y:y+h , x:x+y] = sticker

print(faces)

cv2.imshow('angelina' , image_gray)
cv2.waitKey()
