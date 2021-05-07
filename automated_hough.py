#Hough Transform To Detect Circular Shapes

import numpy as np
import cv2 as cv
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import os


def hough_circle(impath):
    img = cv.imread(impath)
    output = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    edges = cv.Canny(gray,50,150,apertureSize = 3)
    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, 70,
                              param1=50, param2=30, minRadius=0, maxRadius=50)
    detected_circles = np.uint16(np.around(circles)) # its a list of circle parameters (x, y ,radius)
    for (x, y ,r) in detected_circles[0, :]:
        cv.circle(output, (x, y), r, (255, 0, 0), 3)
        cv.circle(output, (x, y), 0, (0, 255, 0), 3)
        
    return output, detected_circles # output is the orig image with cirlcles drawn on it


directory = r'/Users/mohsen/Downloads' # path to the images
labels = []
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        annotated_img, circles = hough_circle(os.path.join(directory,filename))
        labels.append([filename, annotated_img, circles, len(circles[0, :])]) 


# figure(figsize=(10, 10))
# plt.imshow(labels[3][1], interpolation='none', aspect='auto')
# plt.show()



