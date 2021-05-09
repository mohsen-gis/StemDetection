# -*- coding: utf-8 -*-


# Crop images to stem area



# Dependencies

import time
start_time = time.time()

import os

import numpy as np

import matplotlib.pyplot as plt

from skimage import io, feature, filters, color, util, morphology, exposure, segmentation, img_as_float
from skimage.filters import unsharp_mask
from skimage.measure import label, regionprops, perimeter, find_contours
from skimage.morphology import medial_axis, skeletonize, convex_hull_image, binary_dilation, black_tophat, diameter_closing, area_opening, erosion, dilation, opening, closing, white_tophat, reconstruction, convex_hull_object, binary_erosion
from skimage.transform import rescale, resize, downscale_local_mean, rotate
from skimage.util import invert

from glob import glob

import imageio as iio

import scipy
from scipy import ndimage as ndi

from PIL import Image, ImageEnhance 

import cv2 as cv

import math

import pandas as pd                          

# Random label cmap
import matplotlib
import colorsys
def random_label_cmap(n=2**16):

    h,l,s = np.random.uniform(0,1,n), 0.4 + np.random.uniform(0,0.6,n), 0.2 + np.random.uniform(0,0.8,n)
    cols = np.stack([colorsys.hls_to_rgb(_h,_l,_s) for _h,_l,_s in zip(h,l,s)],axis=0)
    cols[0] = 0
    return matplotlib.colors.ListedColormap(cols)

lbl_cmap = random_label_cmap()




#-----------------------------------------------------#
#               Hough Transform
#-----------------------------------------------------#


def hough_circle(img, min_dist, max_radius):
    output = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # img2 = np.floor(img)
    # img2 = img2.astype(int)
    gray = cv.medianBlur(gray, 5)
    edges = cv.Canny(gray,50,150,apertureSize = 3)
    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, min_dist,
                              param1=50, param2=30, minRadius=0, maxRadius=max_radius)
    detected_circles = np.uint16(np.around(circles)) # its a list of circle parameters (x, y ,radius)
    for (x, y ,r) in detected_circles[0, :]:
        cv.circle(output, (x, y), r, (255, 0, 0), -1)
        # cv.circle(output, (x, y), 0, (0, 255, 0), 3)
        
    return output, detected_circles # output is the orig image with cirlcles drawn on it


test_hc, test_dc = hough_circle(img, 70, 40)

red = test_hc[:, :, 0]
# green = image1[:, :, 1]
# blue = image1[:, :, 2]

# Threshold based on the red channel (this depends on the image's background)
bw0 = test_hc[:, :, 0] == 255
plt.imshow(bw0)

labeled_stems, num_stem = label(bw0, return_num = True)
plt.imshow(labeled_stems, cmap = lbl_cmap)


# Dilate
eroded = opening(bw0, out=None)
# plt.imshow(eroded, cmap = 'gray')



distance = ndi.distance_transform_edt(bw0)
   # io.imshow(distance)
   # local_maxi = feature.peak_local_max(distance, indices=False, footprint=morphology.diamond(30), labels=myspk_rot)
local_maxi = feature.peak_local_max(distance, indices=False, min_distance=10, labels=bw0)
# stem = morphology.remove_small_objects(local_maxi, min_size=5)
# io.imshow(img_as_float(local_maxi) - img_as_float(stem))
   
# new_local_max = img_as_float(local_maxi) - img_as_float(stem)
# new_local_max = new_local_max.astype(np.bool)
   
 # local_maxi = feature.corner_peaks(distance, indices=False, min_distance=20, labels=myspk_rot)
   # io.imshow(new_local_max)
   
   
   
markers = ndi.label(local_maxi)[0]
labeled_spikelets = segmentation.watershed(-distance, markers, mask=bw0)
plt.imshow(labeled_spikelets)

regions_spikelets = regionprops(labeled_spikelets)

# n_Spikelets = int(labeled_spikelets[:,:].max())

fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(bw0, cmap=plt.cm.gray)
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap=plt.cm.gray)
ax[1].set_title('Distances')
ax[2].imshow(labeled_spikelets, cmap=lbl_cmap)
ax[2].set_title('Separated objects')

fig.tight_layout()
plt.show()








# Determine regions properties
regions = regionprops(labeled_spks)


#-----------------------------------------------------#
#               Cropping Function
#-----------------------------------------------------#

def CropStems(rgb):
    
    # Read image
    rgb_name = rgb
    rgb = iio.imread(rgb_name)
    
    
    # rgb to gray
    gray0 = rgb @ [0.2126, 0.7152, 0.0722]
    
    # Normalize
    gray0 = gray0/255
    
    # Detect edges
    edges = feature.canny(gray0, sigma = 0.9, low_threshold = 0, high_threshold = .75)
    # plt.imshow(edges, cmap = 'gray')
    
    # Dilate
    dilated = binary_dilation(edges, selem=morphology.diamond(10), out=None)
    # plt.imshow(dilated, cmap = 'gray')
    
    # Get convex hull
    chull = convex_hull_object(dilated, connectivity=2)
    # plt.imshow(chull)
    
    # Apply mask to gray
    new_rgb = np.asarray(chull)
    # gray1 = np.where(chull, gray0, 0)
    new_rgb = np.where(chull[..., None], rgb, 0)
    
    # Crop image
    [rows, columns] = np.where(chull)
    row1 = min(rows)
    row2 = max(rows)
    col1 = min(columns)
    col2 = max(columns)
    cropped = gray1[row1:row2, col1:col2]
    # plt.imshow(cropped)
    
    
    # Verify folder exists
    if os.path.isdir('./CroppedStems') == False:
        os.mkdir("CroppedStems")
    
    # image name
    # cropped_name = "cropped_" + rgb_name
    cropped_name = rgb_name.split("\\")[-1]
    cropped_name = cropped_name.split(".")[-2]
    cropped_name = ".\CroppedStems\\" + cropped_name + "_cropped.JPG"
    
    # Save image
    im = cropped*255
    im = Image.fromarray(im)
    im = im.convert("L")
    im.save(cropped_name)
    
    # return gray1
    








#-----------------------------------------------------#
#               Executing the function
#-----------------------------------------------------#
#             VERIFY FOLDER AND EXTENSION!!!
#-----------------------------------------------------#

# Images folder (change extension if needed)
Images = io.ImageCollection(r'.\Sample_Images\*.JPG')

# Loop through images in folder
for i in Images.files:
    
    # Set the initial time per image
    image_time = time.time()
    
    # Read image
    rgb = i
    
    # Crop image and convert to gray
    CropStems(rgb)
    
    # How long did it take to run this image?
    print("The image", i.split('\\')[-1], "took", time.time() - image_time, "seconds to run.")
    



# How long did it take to run the whole code?
print("This entire code took", time.time() - start_time, "seconds to run.")

    


