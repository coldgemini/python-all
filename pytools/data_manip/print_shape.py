import os
import nrrd
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src image dir")
parser.add_argument("-n", "--n_jobs", type=int, default=3, help="parallel jobs")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
args = parser.parse_args()

filelist = os.listdir(args.src)


def print_shape(filename):
    srcpath = os.path.join(args.src, filename)
    data, _ = nrrd.read(srcpath)
    print(filename, data.shape)


if args.parallel:
    Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(
        delayed(print_shape)(filename) for filename in filelist)
else:
    for filename in filelist:
        print_shape(filename)
