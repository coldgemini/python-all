import numpy as np
import nrrd
from scipy.ndimage import zoom
from xyz_utils.image_utils.image_utils import scale_range

DATA_PATH = '/home/xiang/mnt/Tmp/Debug/aorta/lungcrop.nrrd'
BIN_PATH = '/home/xiang/Tmp/opengl/ct_data.bin'
outshape = (800, 600, 20)

data, _ = nrrd.read(DATA_PATH)
data_np = scale_range(data, -1000, 1000, 1, 254)
data = data.astype(np.uint8)
datashape = data.shape

data = zoom(data, (outshape[0] / datashape[0], outshape[1] / datashape[1], outshape[2] / datashape[2]))

print(data.shape)

data.tofile(BIN_PATH)
