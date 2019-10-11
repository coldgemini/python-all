"""
slice multiple layers in nrrd stack into npz files
"""
import numpy as np
import os
from pypacks.math.range_translation import scale_range
import nrrd
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src image dir")
parser.add_argument("--srcmask", type=str, help="src image dir")
parser.add_argument("--npz", type=str, help="dst npz slice dir")
parser.add_argument("--npzmask", type=str, help="dst npz mask slice dir")
parser.add_argument("-l", "--num_layers", type=int, default=2, help="dst npz mask slice dir")
parser.add_argument("-n", "--n_jobs", type=int, default=2, help="parallel jobs")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
parser.add_argument("-v", "--verbose", action='store_true', default=False, help="verbose")
args = parser.parse_args()


def crop_along_cline(filename):
    if args.verbose:
        print(filename)
    basename = os.path.splitext(filename)[0]
    srcpath = os.path.join(args.src, filename)
    srcmask_path = os.path.join(args.srcmask, filename)

    try:
        srcdata, _ = nrrd.read(srcpath)
    except Exception as e:
        print(str(e))
        print("error: ", srcpath)

    try:
        srcmsk_data, _ = nrrd.read(srcmask_path)
    except Exception as e:
        print(str(e))
        print("error: ", srcmask_path)

    srcdata = scale_range(srcdata, -1000, 1000, 1, 254)
    srcdata = srcdata.astype(np.uint8)
    srcmsk_data = srcmsk_data.astype(np.uint8)

    # for num_skip in range(1, args.num_skip):
    num_layers = args.num_layers
    for img_idx in range(0, num_layers):
        slice_name = basename + '_slice_' + "{0:03d}".format(img_idx) + '_layers_' + "{0}".format(
            num_layers) + '.npz'
        # npz_sub_folder_path = os.path.join(args.npz, basename)
        # npzmask_sub_folder_path = os.path.join(args.npzmask, basename)
        # os.makedirs(npz_sub_folder_path, exist_ok=True)
        # os.makedirs(npzmask_sub_folder_path, exist_ok=True)
        # slice_path = os.path.join(npz_sub_folder_path, slice_name)
        # slice_mask_path = os.path.join(npzmask_sub_folder_path, slice_name)
        slice_path = os.path.join(args.npz, slice_name)
        slice_mask_path = os.path.join(args.npzmask, slice_name)
        npz_slice = srcdata[:, :, img_idx]
        np.savez(slice_path, npz_slice)
        npzmask_slice = srcmsk_data[:, :, img_idx]
        np.savez(slice_mask_path, npzmask_slice)


filelist = os.listdir(args.src)

if args.parallel:
    Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(
        delayed(crop_along_cline)(filename) for filename in filelist)
else:
    for filename in filelist:
        crop_along_cline(filename)
