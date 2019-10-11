import numpy as np
import os
# import nrrd
import SimpleITK as sitk
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mask1", type=str, help="src image dir")
parser.add_argument("--mask2", type=str, help="dst npz slice dir")
parser.add_argument("-o", "--outmask", type=str, help="dst npz slice dir")
parser.add_argument("-n", "--n_jobs", type=int, default=2, help="parallel jobs")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
args = parser.parse_args()

mask1_dir = args.mask1
mask2_dir = args.mask2
outmask_dir = args.outmask
mask1_list = os.listdir(mask1_dir)
mask2_list = os.listdir(mask2_dir)
common_list = list(set(mask1_list).intersection(mask2_list))


def combine_masks(filename):
    print(filename)
    mask1path = os.path.join(mask1_dir, filename)
    mask2path = os.path.join(mask2_dir, filename)
    outmaskpath = os.path.join(outmask_dir, filename)

    # mask1, _ = nrrd.read(mask1path)
    # mask2, _ = nrrd.read(mask2path)
    mask1 = sitk.GetArrayFromImage(sitk.ReadImage(mask1path)).transpose()
    mask2 = sitk.GetArrayFromImage(sitk.ReadImage(mask2path)).transpose()
    assert mask1.shape == mask2.shape

    outmask = np.logical_or(mask1, mask2).astype(np.uint8)

    # nrrd.write(outmaskpath, outmask)
    sitk.WriteImage(sitk.GetImageFromArray(outmask.transpose()), outmaskpath)


if args.parallel:
    Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(
        delayed(combine_masks)(filename) for filename in common_list)
else:
    for filename in common_list:
        combine_masks(filename)
