#Final

import numpy as np
import cv2 as cv
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import os
import os.path


def highpass(img, sigma):
    return img - cv.GaussianBlur(img, (0,0), sigma) + 127


def unsharp_mask(impath, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    image = impath
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def sharpen(impath):
    
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = impath
    return cv.filter2D(img, -1, kernel)


def normalize(array, a, b):
    """
    normalized numpy array into scale [a,b]
    """
    np.seterr(divide="ignore", invalid="ignore")
    return (b-a)*((array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))) + a


# impath = '/Users/mohsen/Desktop/EveryField/LBP/1060.JPG'

sample = cv.imread('/Users/mohsen/Downloads/IWG_stem_img/cropped/1061_cropped.JPG')

dim = sample.shape

def hough_circle(impath, min_dist, max_radius):

    output = cv.imread(impath)
#     output = cv.resize(output, (2500,2260), interpolation = cv.INTER_AREA)
    gray = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
#     gray = highpass(output, 3)
    gray = cv.medianBlur(gray, 3)
    gray = unsharp_mask(gray)
#     gray = sharpen(gray)
#     gray = cv.medianBlur(gray, 3)
    gray = cv.Canny(gray, 50, 150, apertureSize = 3)

    zer = np.zeros(output.shape)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, min_dist,
                              param1=50, param2=30, minRadius=0, maxRadius=max_radius)
    detected_circles = np.uint16(np.around(circles)) # its a list of circle parameters (x, y ,radius)
    counter = 0
    for (x, y ,r) in detected_circles[0, :]:
        
        
#         color = counter * (255/len(detected_circles[0, :]))
#         color1 = (list(np.random.choice(range(256), size=3)))  
#         color =[int(color1[0]), int(color1[1]), int(color1[2])] 
#         color =[int(counter)/255, 0, 0] 
        counter += 1
        cv.circle(zer, (x, y), r, counter, -1)
#         cv.circle(output, (x, y), 0, (0, 255, 0), -1)
        
    return zer, detected_circles # output is the orig image with cirlcles drawn on it



# directory = r'/Users/mohsen/Downloads/IWG_stem_img' # path to the images
cropped_path = r'/Users/mohsen/Downloads/IWG_stem_img/cropped'
hough_on_cropped = r'/Users/mohsen/Downloads/IWG_stem_img/hough_on_cropped/'

resized_hough = '/Users/mohsen/Downloads/IWG_stem_img/hough_on_cropped/resized_hough/'
mask = '/Users/mohsen/Downloads/IWG_stem_img/hough_on_cropped/resized_hough/masks_resized/mask/'
original = '/Users/mohsen/Downloads/IWG_stem_img/hough_on_cropped/resized_hough/masks_resized/original/'
labels = []


for filename in os.listdir(cropped_path):
#     print(filename)
    try:
        if filename.lower().endswith(".jpg"):
#             print(filename)
            annotated_img, circles = hough_circle(os.path.join(cropped_path, filename), 65, 45)
            labels.append([filename, annotated_img, circles, len(circles[0, :])]) 
            f_name = hough_on_cropped + filename[:-4] + '_hough.jpg'
            f_out = mask + filename[:-4] + '_hough.jpg'
            orig_f_out = original + filename[:-4] + '_hough.jpg'
            
            if os.path.isfile(f_name) == True and os.path.isfile(f_out) == False:
                print(f_out)

                cv.imwrite(f_out, annotated_img)
                cv.imwrite(orig_f_out, cv.imread(os.path.join(cropped_path, filename)))
    except:
        continue
        
