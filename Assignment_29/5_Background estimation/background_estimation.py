import cv2
import numpy as np

cap = cv2.VideoCapture('5_Background estimation/input/cars.mp4')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
avg_frame = np.zeros((frame_height, frame_width, 3), dtype=np.float32)

count = 0
while count < total_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    avg_frame += frame.astype(np.float32)  
    count += 1
    
avg_frame /= count
    
avg_frame = cv2.convertScaleAbs(avg_frame)
    
cap.release()
    
cv2.imwrite('5_Background estimation/output/empty_road.png', avg_frame)
cv2.imshow('Background Image', avg_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
