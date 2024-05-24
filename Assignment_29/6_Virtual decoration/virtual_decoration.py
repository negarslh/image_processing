import cv2

room_image = cv2.imread('6_Virtual decoration/input/room.jpg')
floor_image = cv2.imread('6_Virtual decoration/input/parket.jpg')
mask = cv2.imread('6_Virtual decoration/input/mask.jpg', cv2.IMREAD_GRAYSCALE)

floor_image_resized = cv2.resize(floor_image, (room_image.shape[1], room_image.shape[0]))

floor_image_masked = cv2.bitwise_and(floor_image_resized, floor_image_resized, mask=mask)

mask_inverted = cv2.bitwise_not(mask)

room_image_masked = cv2.bitwise_and(room_image, room_image, mask=mask_inverted)

final_image = cv2.add(floor_image_masked, room_image_masked)

cv2.imshow('Virtual Decoration', final_image)
cv2.waitKey()
cv2.imwrite('6_Virtual decoration/output/result.jpg', final_image)
