from time import time
import nrrd
import numpy as np
import scipy.ndimage as ndimage
from skimage.transform import SimilarityTransform, warp


def rotate(image, angle, cval):
    center = np.array((image.shape[0], image.shape[1])) / 2. - 0.5
    tform1 = SimilarityTransform(translation=center)
    tform2 = SimilarityTransform(rotation=angle * np.pi / 180)
    tform3 = SimilarityTransform(translation=-center)
    tform = tform3 + tform2 + tform1

    result = []
    for i in range(image.shape[2]):
        result.append(warp(image[:, :, i], tform, cval=cval))
    return np.transpose(result, (1, 2, 0))


# data_path = '/data2/home/zhouxiangyong/Data/aorta_seg_data/coronary/scalar_raw/1.2.156.112605.14038013507713.181220010026.3.MMDhyLhmbT8ZIvGcdrIBEhRoHGyIVN.nrrd'
data_path = '/data2/home/zhouxiangyong/Tmp/cuda_rotate3d_input.nrrd'
data_sk_path = '/data2/home/zhouxiangyong/Tmp/rotate_sk.nrrd'
data_nd_path = '/data2/home/zhouxiangyong/Tmp/rotate_nd.nrrd'

data, _ = nrrd.read(data_path)

input("to start rotating?")
print("rotating by skimage ...")
start_t = time()
data_sk = rotate(data, 45, 0)
end_t = time()
print("rotate time: {}".format(end_t - start_t))

print("rotating by ndimage ...")
start_t = time()
data_nd = ndimage.rotate(data, angle=45, axes=(0, 1),
                         reshape=False, order=0, mode='nearest', cval=0)
end_t = time()
print("rotate time: {}".format(end_t - start_t))

nrrd.write(data_sk_path, data_sk)
nrrd.write(data_nd_path, data_nd)
