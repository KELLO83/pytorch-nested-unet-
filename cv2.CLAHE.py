import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
image_file = 'inputs/CRACKTREE200_INPUT/images/6193.jpg'
image = cv2.imread(image_file, cv2.IMREAD_COLOR)


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

eq_image = cv2.equalizeHist(gray_image)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(gray_image)

cv2.imshow("eq",eq_image)
cv2.imshow("clahe_image",clahe_image)
cv2.waitKey(0)
