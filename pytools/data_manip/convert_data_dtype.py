import numpy as np
import os
import nrrd
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src image dir")
parser.add_argument("-d", "--dst", type=str, help="dst npz slice dir")
parser.add_argument("-n", "--n_jobs", type=int, default=2, help="parallel jobs")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
args = parser.parse_args()

src_dir = args.src
dst_dir = args.dst
src_list = os.listdir(src_dir)


def convert_dtype(filename):
    print(filename)
    srcpath = os.path.join(src_dir, filename)
    dstpath = os.path.join(dst_dir, filename)

    src_data, _ = nrrd.read(srcpath)
    dst_data = src_data.astype(np.uint8)

    nrrd.write(dstpath, dst_data)


if args.parallel:
    Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(
        delayed(convert_dtype)(filename) for filename in src_list)
else:
    for filename in src_list:
        convert_dtype(filename)
