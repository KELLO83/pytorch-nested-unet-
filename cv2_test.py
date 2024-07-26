import cv2
import matplotlib.pyplot as plt

image_file = 'inputs/CRACKTREE200_INPUT/images/6193.jpg'
image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
image_copy = image.copy()


eq_image = cv2.equalizeHist(image)
eq_image = cv2.cvtColor(eq_image,cv2.COLOR_GRAY2RGB)
# cv2.imshow('origin',image_copy)
# cv2.imshow("test",eq_image)
# cv2.waitKey(0)
# #cv2.destroyAllWindows()

image_copy = image_copy.flatten()
eq_image , _ , _ = cv2.split(eq_image)
eq_image = eq_image.flatten()
plt.subplot(1,2,1)
plt.hist(image_copy , bins=50 , color='red')
plt.subplot(1,2,2)
plt.hist(eq_image, bins=50 , color='blue')
plt.show()