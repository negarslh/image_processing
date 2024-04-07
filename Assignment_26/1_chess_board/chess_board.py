import cv2
import numpy as np

rows = 8
cols = 8
size = 50

chess_board = np.zeros((rows*size , cols*size) , dtype=np.uint8)

for i in range(rows):
    for j in range(cols):
        if (i+j) % 2 == 0 :
            chess_board[i * size:(i+1) * size , j * size:(j+1) * size] = 255

cv2.imshow("chess board" , chess_board)
cv2.waitKey()
cv2.imwrite('chess_board.jpg', chess_board)
