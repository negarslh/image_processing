import cv2
import numpy as np

pitch = np.zeros((500, 700, 3), dtype=np.uint8) 

pitch[:, :, 1] = 100  

cv2.rectangle(pitch, (50, 50), (650, 450), (255, 255, 255), 2)  
cv2.line(pitch, (350, 50), (350, 450), (255, 255, 255), 2)      
cv2.circle(pitch, (350, 250), 100, (255, 255, 255), 2)          
cv2.circle(pitch, (350, 250), 10 , (255, 255, 255), -1)          

cv2.rectangle(pitch, (50, 175), (100, 325), (255, 255, 255), 2)  
cv2.rectangle(pitch, (600, 175), (650, 325), (255, 255, 255), 2) 


cv2.imshow('Football Pitch', pitch)
cv2.waitKey(0)
cv2.destroyAllWindows()
