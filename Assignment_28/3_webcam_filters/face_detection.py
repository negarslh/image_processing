import cv2

cap = cv2.VideoCapture(0)

face_sticker = cv2.imread("3_webcam_filters/input/imoji.webp")  
glasses_sticker = cv2.imread("3_webcam_filters/input/sunglasses.png")
smile_sticker = cv2.imread("3_webcam_filters/input/Lips.png")

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

mode = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if mode == 1:
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_sticker_resized = cv2.resize(face_sticker, (w, h))
            frame[y:y+h, x:x+w] = face_sticker_resized

    elif mode == 2:
        eyes = eye_detector.detectMultiScale(gray)
        if len(eyes) >= 2:
            x1, y1, w1, h1 = eyes[0]
            x2, y2, w2, h2 = eyes[1]
            glasses_width = max(x2 + w2 - x1, w1)
            glasses_height = max(y2 + h2 - y1, h1)
            glasses_sticker_resized = cv2.resize(glasses_sticker, (glasses_width, glasses_height))
            frame[y1:y1+glasses_height, x1:x1+glasses_width] = glasses_sticker_resized

        smiles = smile_detector.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=22)
        for (x, y, w, h) in smiles:
            smile_sticker_resized = cv2.resize(smile_sticker, (w, h))
            frame[y:y+h, x:x+w] = smile_sticker_resized

    cv2.imshow("Face Filters", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('1'):
        mode = 1
    elif key == ord('2'):
        mode = 2

cap.release()
cv2.destroyAllWindows()
