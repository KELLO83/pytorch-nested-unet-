import cv2 
import numpy as np


image_file = 'inputs/CRACKTREE200_INPUT/images/6192.jpg'
image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
image_copy = image.copy()

img_dx = cv2.Sobel(image,cv2.CV_64F, 0 , 1,ksize=3)
img_dy = cv2.Sobel(image,cv2.CV_64F , 1,0 , ksize=3)
img_mag = cv2.magnitude(img_dx , img_dy)
img_mag = cv2.normalize(img_mag , 0, 255, cv2.NORM_MINMAX)
# img_mag = np.uint8(img_mag)

cv2.imshow('dx',img_dx)
cv2.imshow('dy',img_dy)
cv2.imshow('origin',image_copy)
cv2.imshow('test',img_mag)
cv2.waitKey(0)
cv2.destroyAllWindows()
