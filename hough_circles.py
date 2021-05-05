#Hough Transform To Detect Circular Shapes

import numpy as np
import cv2 as cv
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
img = cv.imread('/Users/mohsen/Desktop/EveryField/LBP/cropped1.jpg')
output = img.copy()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.medianBlur(gray, 5)
edges = cv.Canny(gray,50,150,apertureSize = 3)
circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, 70,
                          param1=50, param2=30, minRadius=0, maxRadius=50)
detected_circles = np.uint16(np.around(circles))
for (x, y ,r) in detected_circles[0, :]:
    cv.circle(output, (x, y), r, (255, 0, 0), 3)
    cv.circle(output, (x, y), 0, (0, 255, 0), 3)

print('Stem counts: ', len(detected_circles[0, :]))
figure(figsize=(10, 10))
plt.imshow(output, interpolation='none', aspect='auto')

plt.show()