import numpy as np
import os
import nrrd
import nibabel as nib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src image")
parser.add_argument("-d", "--dst", type=str, help="dst image dir")
parser.add_argument("-l", "--list", type=str, help="cline list")
args = parser.parse_args()
srcpath = args.src
dstdir = args.dst
listfile = args.list

# print(clistfile)
listfile_h = open(listfile, "r")
file_lines = listfile_h.readlines()
file_lines = [line.rstrip() for line in file_lines]
listfile_h.close()

print(srcpath)
# mask, header = nrrd.read(srcpath)
vol_file = nib.load(srcpath)
mask = vol_file.get_data()
mask = mask.astype(np.uint8)
xS, yS, zS = mask.shape
print(xS, yS, zS)
restore_height = 238

layer_height = 49
for idx, filename in enumerate(file_lines):
    dstpath = os.path.join(dstdir, filename)
    print(dstpath)
    dst_mask = np.zeros((xS, yS, restore_height), dtype=np.uint8)
    start_idx = layer_height * idx + 1
    end_idx = layer_height * (idx + 1)
    dst_mask_squeeze = mask[:, :, start_idx:end_idx]
    dst_mask[:, :, 0::5] = dst_mask_squeeze
    nrrd.write(dstpath, dst_mask)
