import skimage.morphology
from scipy import ndimage

skeleton = skimage.morphology.skeletonize_3d(mask0)

ndimage.binary_dilation(a).astype(a.dtype)
