"""
slice multiple layers in nrrd stack into npz files
"""
import numpy as np
from PIL import Image
import cv2
import os
# from pypacks.math.range_translation import scale_range
import nrrd
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src image dir")
parser.add_argument("-d", "--dst", type=str, help="dst npz slice dir")
parser.add_argument("-n", "--n_jobs", type=int, default=2, help="parallel jobs")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
args = parser.parse_args()


def crop_along_cline(filename):
    basename = os.path.splitext(filename)[0]
    srcpath = os.path.join(args.src, filename)
    # dstpath = os.path.join(args.dst, filename)

    try:
        srcdata, _ = nrrd.read(srcpath)
        _, _, zS = srcdata.shape

        srcdata = srcdata.astype(np.uint8)

        # for num_skip in range(1, args.num_skip):
        for img_idx in range(zS):
            slice_name = basename + '_slice_' + "{0:03d}".format(img_idx) + '.png'
            slice_path = os.path.join(args.dst, slice_name)
            slice = srcdata[:, :, img_idx]
            slice = cv2.resize(slice, (256, 256))
            slice = np.fliplr(slice)
            slice = Image.fromarray(slice, mode='L')
            slice.save(slice_path)
    except Exception as e:
        print(str(e))
        print("error: ", srcpath)


filelist = os.listdir(args.src)

if args.parallel:
    Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(
        delayed(crop_along_cline)(filename) for filename in filelist)
else:
    for filename in filelist:
        crop_along_cline(filename)
