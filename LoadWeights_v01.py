

# Loading weights from model








# Random label cmap
import matplotlib
import colorsys
def random_label_cmap(n=2**16):

    h,l,s = np.random.uniform(0,1,n), 0.4 + np.random.uniform(0,0.6,n), 0.2 + np.random.uniform(0,0.8,n)
    cols = np.stack([colorsys.hls_to_rgb(_h,_l,_s) for _h,_l,_s in zip(h,l,s)],axis=0)
    cols[0] = 0
    return matplotlib.colors.ListedColormap(cols)

lbl_cmap = random_label_cmap()



from stardist.models import StarDist2D 
from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from csbdeep.utils import Path, download_and_extract_zip_file, normalize
from skimage import io, feature, filters, color, util, morphology, exposure, segmentation, img_as_float
from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize
import matplotlib.pyplot as plt



# Set path for images and labels
X = sorted(glob('ground_truth/images/*.tif'))
Y = sorted(glob('ground_truth/masks/*.tif'))
assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))

# Load sample image
X, Y = X[:10], Y[:10]

X = list(map(imread,X))
Y = list(map(imread,Y))
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]




# NORMALIZE
# This may not be necessary in our case

axis_norm = (0,1)   # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
    sys.stdout.flush()

X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]










# Should have a structure like this:
#     StemDetection\models\stardist\

model = StarDist2D(None, name='stardist', basedir='models')



# # Images folder (change extension if needed)
# Images = io.ImageCollection(r'.\Sample_Images\*.JPG')

# # Testing on first image
# img_name = Images.files[0]
# img = io.imread(img_name)
# io.imshow(img)

# Normalize
img = normalize(X[9], 1,99.8, axis=axis_norm)

# Predict
labels, details = model.predict_instances(img)

# Plot
plt.figure(figsize=(8,8))
plt.imshow(img if img.ndim==2 else img[...,0], clim=(0,1), cmap='gray')
plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)
plt.axis('off');




