import nrrd
from skimage.morphology import dilation
import numpy as np
from centerlines.centerlines import get_center_lines
from util.util import extend_curve

CLINE_PATH = '/data2/home/zhouxiangyong/Data/aorta_seg_data/cta/single_label_trim4cline_crop_extend_combined/combined_smoothed.nrrd'
CLINE_BASE_PATH = '/data2/home/zhouxiangyong/Tmp/cline_base.nrrd'
CLINE_EXT_PATH = '/data2/home/zhouxiangyong/Tmp/cline_ext.nrrd'
CLINE_BOTH_PATH = '/data2/home/zhouxiangyong/Tmp/cline_both.nrrd'
cline_mask, _ = nrrd.read(CLINE_PATH)
new_mask_base = np.zeros_like(cline_mask, dtype=np.uint8)
new_mask_ext = np.zeros_like(cline_mask, dtype=np.uint8)

NUM_EXT = 35
NUM_FIT = 20
clines = get_center_lines(cline_mask, min_curve_len=50)
cline = clines[0]
# cline_reverse = cline.copy()
# cline_reverse.reverse()
cline_reverse = np.flip(cline, axis=0)
cline_last = cline_reverse[-1 - NUM_FIT:-1]
print(cline_last.shape)
cline_ext = extend_curve(cline_last, NUM_EXT, degree=2)
# cline_ext = np.flip(cline_ext, axis=0)
print(cline_ext.shape)
cline_head = cline_ext[-1 - NUM_EXT:-1]
cline_head = np.flip(cline_head, axis=0)
print(cline_head.shape)
cline_longer = np.concatenate((cline_head, cline), axis=0)
len_cline = cline.shape[0]
len_cline_longer = cline_longer.shape[0]

# print(type(clines))
# print(clines[0].shape)
# print(type(cline_ext))
# print(cline_ext.shape)
#
# print(cline.shape)
# print(cline)

print("for cline")
for idx in range(len_cline):
    x, y, z = tuple(int(elem) for elem in cline[idx])
    print("idx: {0}, x: {1}, y: {2}, z: {3}".format(idx, x, y, z))
    new_mask_base[x, y, z] = 1

skel_base = dilation(new_mask_base)

print("for cline_ext")
for idx in range(len_cline_longer):
    x, y, z = tuple(int(elem) for elem in cline_longer[idx])
    print("idx: {0}, x: {1}, y: {2}, z: {3}".format(idx, x, y, z))
    new_mask_ext[x, y, z] = 1

skel_ext = dilation(new_mask_ext)

skel_both = skel_base + skel_ext

# nrrd.write(CLINE_BASE_PATH, skel_base)
# nrrd.write(CLINE_EXT_PATH, skel_ext)
nrrd.write(CLINE_BOTH_PATH, skel_both)
