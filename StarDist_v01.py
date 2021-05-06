# -*- coding: utf-8 -*-


# # Install TensorFlow
# pip install tensorflow

# # install StarDist
# pip install stardist


from stardist.models import StarDist2D 



# ----------------------------------#
# ------- Pretrained Model ---------#
#-----------------------------------#

# prints a list of available models 
StarDist2D.from_pretrained() 

# creates a pretrained model
model = StarDist2D.from_pretrained('2D_versatile_fluo')

labels, _ = model.predict_instances(img)







# ------------------------------------------#
# ------- Testing Pretrained Model ---------#
#                                       

# Random label cmap
import matplotlib
import colorsys
def random_label_cmap(n=2**16):

    h,l,s = np.random.uniform(0,1,n), 0.4 + np.random.uniform(0,0.6,n), 0.2 + np.random.uniform(0,0.8,n)
    cols = np.stack([colorsys.hls_to_rgb(_h,_l,_s) for _h,_l,_s in zip(h,l,s)],axis=0)
    cols[0] = 0
    return matplotlib.colors.ListedColormap(cols)

lbl_cmap = random_label_cmap()




from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean

# Gather the image files (change path)
Images = io.ImageCollection(r'.\Test_Images\*.JPG')


# Read and visualize the image
rgb0 = Images[0]
# plt.imshow(rgb0)

# Convert to gray
gray0 = rgb0 @ [0.2126, 0.7152, 0.0722]
plt.imshow(gray0, cmap = 'gray')

import cv2 as cv
gray0 = cv.Canny(rgb0, 50, 150, apertureSize=3)
plt.imshow(gray0, cmap='gray')

gray0 = gray0.astype('float64')

gray1 = rescale(gray0, 1, anti_aliasing=False)
plt.imshow(gray1, cmap = 'gray')

# img_adapteq = exposure.equalize_adapthist(gray0, clip_limit=0.03)
# plt.imshow(img_adapteq, cmap='gray')

labels, _ = model.predict_instances(gray1)
plt.imshow(labels, cmap = lbl_cmap)



img, lbl = X_small[i], fill_label_holes(Y_small[i])



















# ----------------------------------#
# ---------- Custom Model ----------#
#-----------------------------------#

# Requires annotated images (training data)

# Followed this tutorial: https://github.com/maweigert/neubias_academy_stardist/blob/master/notebooks/stardist_example_2D_colab.ipynb
# Video: https://www.youtube.com/watch?v=Amn_eHRGX5M&t=1874s



import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, download_and_extract_zip_file, normalize

from stardist.matching import matching_dataset
from stardist import fill_label_holes, random_label_cmap, relabel_image_stardist, calculate_extents, gputools_available, _draw_polygons
from stardist.models import Config2D, StarDist2D, StarDistData2D


np.random.seed(42)
lbl_cmap = random_label_cmap()



# ---------------------------------------------
#               1. The Data
#----------------------------------------------


# Download sample data
download_and_extract_zip_file(
    url       = 'https://github.com/mpicbg-csbd/stardist/releases/download/0.1.0/dsb2018.zip',
    targetdir = 'data',
    verbose   = 1,
)


!find data -type d | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"


fX = sorted(Path('data/dsb2018/train/images/').glob('*.tif'))
fY = sorted(Path('data/dsb2018/train/masks').glob('*.tif'))
print(f"found {len(fX)} training images and {len(fY)} training masks")
assert all(Path(x).name==Path(y).name for x,y in zip(fX,fY))

# Load a small subset for display
fX_small, fY_small = fX[:10], fY[:10]

X_small = list(map(imread,map(str,fX_small)))
Y_small = list(map(imread,map(str,fY_small)))


# Example image
i = min(4, len(X_small)-1)
img, lbl = X_small[i], fill_label_holes(Y_small[i])
assert img.ndim in (2,3)
img = img if img.ndim==2 else img[...,:3]
# assumed axes ordering of img and lbl is: YX(C)



plt.figure(figsize=(10,8))
plt.subplot(121); plt.imshow(img,cmap='gray');   plt.axis('off'); plt.title('Raw image')
plt.subplot(122); plt.imshow(lbl,cmap=lbl_cmap, interpolation="nearest"); plt.axis('off'); plt.title('GT labels (mask)')
None;


# Fitting ground-truth labels with star-convex polygons
n_rays = [2**i for i in range(2,8)]
print(n_rays)


# Example image reconstructed with various number of rays
fig, ax = plt.subplots(2,3, figsize=(12,8))
for a,r in zip(ax.flat,n_rays):
    a.imshow(relabel_image_stardist(lbl, n_rays=r), cmap=lbl_cmap, interpolation="nearest")
    a.set_title('Reconstructed (%d rays)' % r)
    a.axis('off')
plt.tight_layout();

# The more rays the better the approximmation to the rounded shape
# We problably won't need more than 16?



# Mean IoU (Intersection over union?) for different number of rays (measure of accuracy)
scores = []
for r in tqdm(n_rays):
    Y_reconstructed = [relabel_image_stardist(lbl, n_rays=r) for lbl in Y_small]
    mean_iou = matching_dataset(Y_small, Y_reconstructed, thresh=0, show_progress=False).mean_true_score
    scores.append(mean_iou)

plt.figure(figsize=(8,5))
plt.plot(n_rays, scores, 'o-')
plt.xlabel('Number of rays for star-convex polygon')
plt.ylabel('Reconstruction score (mean IoU)')
plt.title("Mean IoU of ground truth reconstruction (should be > 0.8 for a reasonable number of rays)")
None;





# ---------------------------------------------
#               2. Training
#----------------------------------------------


# Read all images from training
fX = sorted(Path('data/dsb2018/train/images/').glob('*.tif'))
fY = sorted(Path('data/dsb2018/train/masks').glob('*.tif'))
assert all(Path(x).name==Path(y).name for x,y in zip(fX,fY))
print(f"{len(fX)} files found")

X = list(map(imread,map(str,tqdm(fX))))
Y = list(map(imread,map(str,tqdm(fY))))
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]


# Normalize them
axis_norm = (0,1)   # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
    sys.stdout.flush()

X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]



# Split into train and validation datasets.
assert len(X) > 1, "not enough training data"
rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1, int(round(0.15 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] 
print('number of images: %3d' % len(X))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))



# Training data consists of pairs of input image and label instances.
i = min(9, len(X)-1)
img, lbl = X[i], Y[i]
assert img.ndim in (2,3)
img = img if img.ndim==2 else img[...,:3]
plt.figure(figsize=(10,8))
plt.subplot(121); plt.imshow(img,cmap='gray');   plt.axis('off'); plt.title('Raw image')
plt.subplot(122); plt.imshow(lbl,cmap=lbl_cmap, interpolation="nearest"); plt.axis('off'); plt.title('GT labels')
None;






# ---------------------------------------------
#               3. Configuration
#----------------------------------------------

# A StarDist2D model is specified via a Config2D object.
print(Config2D.__doc__)     # We should change some of these only if we need to


conf = Config2D (
    n_rays       = 32,
    grid         = (2,2),   # sampling every second pixel?
    n_channel_in = 1,
)
print(conf)
vars(conf)
# Note: The trained StarDist2D model will not predict completed shapes for partially visible objects at the image boundary if train_shape_completion=False (which is the default option).

# name the model
model = StarDist2D(conf, name='stardist', basedir='models')


# Check if the neural network has a large enough field of view to see up to the boundary of most objects.

median_size = calculate_extents(list(Y), np.median)
fov = np.array(model._axes_tile_overlap('YX'))
if any(median_size > fov):
    print("WARNING: median object size larger than field of view of the neural network.")
else:
    print("All good! (object sizes fit into field of view of the neural network)")




# The following command can be used to show a list of available pretrained models:
StarDist2D.from_pretrained()



# ---------------------------------------------
#               Augmentation
#----------------------------------------------

# You can define a function/callable that applies augmentation to each batch of the data generator.
# We here use an augmenter that applies random rotations, flips, and intensity changes, which are typically sensible for (2D) microscopy images (not sure if this will be necessary for us):

def random_fliprot(img, mask): 
    axes = tuple(range(img.ndim)) 
    perm = np.random.permutation(axes)
    img = img.transpose(perm) 
    mask = mask.transpose(perm) 
    for ax in axes: 
        if np.random.rand()>.5:
            img = np.flip(img,axis = ax)
            mask = np.flip(mask,axis = ax)
    return img, mask 

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-.2,.2)
    return img


def augmenter(img,mask):
    """Augmentation for image,mask"""
    img, mask = random_fliprot(img, mask)
    img = random_intensity_change(img)
    return img, mask



plt.figure(figsize=(8,5))
plt.subplot(121); plt.imshow(img,cmap='gray', clim = (0,1));   plt.axis('off'); plt.title('Raw image')
plt.subplot(122); plt.imshow(lbl,cmap=lbl_cmap, interpolation="nearest"); plt.axis('off'); plt.title('GT labels (mask)')
  
for _ in range(4):
    plt.figure(figsize=(8,5))
    x,y = augmenter(img, lbl)
    plt.subplot(121); plt.imshow(x,cmap='gray', clim = (0,1));   plt.axis('off'); plt.title('Augmented: Raw image')
    plt.subplot(122); plt.imshow(y,cmap=lbl_cmap, interpolation="nearest"); plt.axis('off'); plt.title('Augmented: GT labels (mask)')
None;




# ---------------------------------------------
#               Model Training
#----------------------------------------------

# We recommend to monitor the progress during training with TensorBoard: https://www.tensorflow.org/guide.

!rm -rf logs

%reload_ext tensorboard
%tensorboard --logdir=. --port 6008


quick_demo = True

if quick_demo:
    print (
        "NOTE: This is only for a quick demonstration!\n"
        "      Please set the variable 'quick_demo = False' for proper (long) training.",
        file=sys.stderr, flush=True
    )
    model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter,
                epochs=40, steps_per_epoch=25)

    print("====> Stopping training and loading previously trained demo model from disk.", file=sys.stderr, flush=True)
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
else:
    model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter)
None;




# ---------------------------------------------
#               Threshold optimization
#
# While the default values for the probability and non-maximum suppression thresholds already yield good results in many cases, we still recommend to adapt the thresholds to your data. The optimized threshold values are saved to disk and will be automatically loaded with the model.

if not quick_demo:
    model.optimize_thresholds(X_val, Y_val)






# ---------------------------------------------
#               3. Prediction
#----------------------------------------------


# We now load images from the sub-folder test that have not been used during training.

fXt = sorted(Path('data/dsb2018/test/images/').glob('*.tif'))
print(f"{len(fXt)} files found")
Xt = list(map(imread,map(str,tqdm(fXt))))

n_channel = 1 if Xt[0].ndim == 2 else Xt[0].shape[-1]
axis_norm = (0,1)   # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))


# Prediction

# Make sure to normalize the input image beforehand or supply a normalizer to the prediction function.

img = normalize(Xt[16], 1,99.8, axis=axis_norm)
labels, details = model.predict_instances(img, verbose = True)

plt.figure(figsize=(5,5))
plt.imshow(img if img.ndim==2 else img[...,:3], clim=(0,1), cmap='gray')
plt.imshow(labels, cmap=lbl_cmap, interpolation="nearest", alpha=0.5)
plt.axis('off');



# More example results

def example(model, i, show_dist=True):
    img = normalize(Xt[i], 1,99.8, axis=axis_norm)
    labels, details = model.predict_instances(img)

    plt.figure(figsize=(13,10))
    img_show = img if img.ndim==2 else img[...,:3]
    coord, points, prob = details['coord'], details['points'], details['prob']
    plt.subplot(121); plt.imshow(img_show, cmap='gray'); plt.axis('off')
    a = plt.axis()
    _draw_polygons(coord, points, prob, show_dist=show_dist)
    plt.axis(a)
    plt.subplot(122); plt.imshow(img_show, cmap='gray'); plt.axis('off')
    plt.imshow(labels, cmap=lbl_cmap, interpolation="nearest", alpha=0.5)
    plt.tight_layout()
    plt.show()


example(model, 42)
example(model, 1)
example(model, 15)


# Evaluation and Detection Performance

# First predict the labels for all validation images:
Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
              for x in tqdm(X_val)]

# Choose several IoU thresholds $\tau$ that might be of interest and for each compute matching statistics for the validation data.
taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
stats = [matching_dataset(Y_val, Y_val_pred, thresh=t, show_progress=False) for t in tqdm(taus)]

# Example: Print all available matching statistics for $\tau=0.5$
stats[taus.index(0.5)]


# Plot the matching statistics and the number of true/false positives/negatives as a function of the IoU threshold $\tau$.
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))

for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score'):
    ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
ax1.set_xlabel(r'IoU threshold $\tau$')
ax1.set_ylabel('Metric value')
ax1.grid()
ax1.legend()

for m in ('fp', 'tp', 'fn'):
    ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
ax2.set_xlabel(r'IoU threshold $\tau$')
ax2.set_ylabel('Number #')
ax2.grid()
ax2.legend();



# Export model to Fiji if desired
if not quick_demo:
    model.export_TF()