import numpy as np
import nrrd
import scipy.ndimage
from scipy.ndimage.filters import maximum_filter
# from scipy.ndimage.morphology import generate_binary_structure
from skimage.morphology import ball, cube

heatmap_path = '/home/xiang/mnt/Workspace/Dev/semantic-seg/datasets/inference/unet3D_heatmap_long/heatmap/1.2.840.113619.2.55.3.2831164355.488.1510306320.268.4.nrrd'
output_path = '/home/xiang/Tmp/output.nrrd'

heatmap, _ = nrrd.read(heatmap_path)
print(heatmap.shape)
print(heatmap.dtype)

gauss = scipy.ndimage.filters.gaussian_filter(heatmap, 3, truncate=5)
thresh = gauss > 0.2
# neighborhood = generate_binary_structure(2, 2, 2)
# neighborhood = cube(3)
# local_max = maximum_filter(gauss, footprint=neighborhood) == gauss
local_max = maximum_filter(gauss, size=6) == gauss
filtered_max = np.logical_and(local_max, thresh)
filtered_max = filtered_max.astype(np.int32)

# nrrd.write(output_path, gauss)
nrrd.write(output_path, filtered_max)
