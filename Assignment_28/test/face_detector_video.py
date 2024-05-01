import cv2

cap = cv2.VideoCapture(0)
image_sticker = cv2.imread('test/sticker.png')
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    _ , frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_detector.detectMultiScale(frame)

    for face in faces:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        sticker = cv2.resize(image_sticker , [w,h])
        frame[y:y+h, x:x+w] = sticker

    cv2.imshow('result' , frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    