import numpy as np
from skimage.morphology import ball
from skimage.morphology import binary_dilation
import os
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


def dilate(filename):
    print(filename)
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_dir, filename)

    mask, _ = nrrd.read(src_path)

    selem = ball(4)
    mask = binary_dilation(mask, selem).astype(np.uint8)

    nrrd.write(dst_path, mask)


src_list = os.listdir(src_dir)

if args.parallel:
    Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(
        delayed(dilate)(filename) for filename in src_list)
else:
    for filename in src_list:
        dilate(filename)
