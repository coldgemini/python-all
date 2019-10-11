import numpy as np
import os
from skimage.measure import label
import nrrd
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src image dir")
parser.add_argument("-d", "--dst", type=str, help="dst slice dir")
parser.add_argument("-n", "--n_jobs", type=int, default=2, help="parallel jobs")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
args = parser.parse_args()

src_dir = args.src
dst_dir = args.dst


def getLargestCC_helper(segmentation):
    labels = label(segmentation)
    unique, counts = np.unique(labels, return_counts=True)
    list_seg = list(zip(unique, counts))[1:]  # the 0 label is by default background so take the rest
    largest = max(list_seg, key=lambda x: x[1])[0]
    labels_max = (labels == largest).astype(np.uint8)
    return labels_max


def get_largest_CC(filename):
    print(filename)
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_dir, filename)

    mask, _ = nrrd.read(src_path)
    mask_bin = np.where(mask != 0, 1, 0)

    mask_bin_lcc = getLargestCC_helper(mask_bin).astype(np.uint8)

    mask_out = np.where(mask_bin_lcc == 0, 0, mask)

    nrrd.write(dst_path, mask_out)


src_list = os.listdir(src_dir)

if args.parallel:
    Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(
        delayed(get_largest_CC)(filename) for filename in src_list)
else:
    for filename in src_list:
        get_largest_CC(filename)
