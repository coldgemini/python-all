import numpy as np
import nrrd
from coordinates.coords import calc_coords_single_blob
from image_utils.heatmap import gen_heatmap_3d

data_path = '/home/xiang/mnt/Workspace/Dev/AortaSlice/data/tmp/data_to_tune_model/data/1.2.840.113619.2.55.3.2831164355.488.1510306320.268.4.nrrd'
label_path = '/home/xiang/mnt/Workspace/Dev/AortaSlice/data/tmp/data_to_tune_model/label/1.2.840.113619.2.55.3.2831164355.488.1510306320.268.4.nrrd'
heatmap_path = '/home/xiang/Tmp/heatmap/test1.nrrd'

data, _ = nrrd.read(data_path)
label, _ = nrrd.read(label_path)
print(label.shape)

l2_coords = calc_coords_single_blob(label == 2)
l3_coords = calc_coords_single_blob(label == 3)
l4_coords = calc_coords_single_blob(label == 4)

print(l2_coords)

heatmap = np.zeros_like(label, dtype=np.float32)
heatmap_l2 = gen_heatmap_3d(heatmap.shape, l2_coords, 4)
heatmap_l3 = gen_heatmap_3d(heatmap.shape, l3_coords, 4)
heatmap_l4 = gen_heatmap_3d(heatmap.shape, l4_coords, 4)

print(heatmap_l2.dtype)
print(heatmap_l2.shape)
print(heatmap_l3.dtype)
print(heatmap_l4.dtype)

heatmap = np.maximum.reduce([heatmap_l2, heatmap_l3, heatmap_l4])

nrrd.write(heatmap_path, heatmap)
