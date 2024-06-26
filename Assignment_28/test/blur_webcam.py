import cv2

cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


while True:
    _ , frame = cap.read()
    faces = face_detector.detectMultiScale(frame)

    for face in faces :
        x,y,w,h = face

        face_image = frame[y:y+h , x:x+w]
        face_image_small = cv2.resize(face_image , [15,15])
        face_image_big = cv2.resize(face_image_small , [w,h] , interpolation = cv2.INTER_NEAREST)
        frame[y:y+h , x:x+w] = face_image_big

        cv2.imshow('result' , frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

