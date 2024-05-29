import cv2

def focus_on_flower(image, blur_strength=21, threshold_value=200):
    
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Unable to read the image. Check the path and try again.")
    
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    _, mask = cv2.threshold(blurred_image, threshold_value, 255, cv2.THRESH_BINARY)
    flower = cv2.bitwise_and(image, image, mask=mask)
    mask_inv = cv2.bitwise_not(mask)

    background = cv2.bitwise_and(image, image, mask=mask_inv)
    blurred_background = cv2.GaussianBlur(background, (blur_strength, blur_strength), 0)
    final_image = cv2.add(flower, blurred_background)
    
    return final_image

image = '2_Foreground focus, Blur background/input/image.png'
result_image = focus_on_flower(image)

cv2.imwrite('2_Foreground focus, Blur background/output/result.png', result_image)
cv2.imshow('Final Image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
