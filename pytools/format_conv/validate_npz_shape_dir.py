import os
import numpy as np
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src image dir")
parser.add_argument('--shape', nargs='+', type=int)
parser.add_argument("-n", "--n_jobs", type=int, default=2, help="parallel jobs")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
parser.add_argument("-v", "--verbose", action='store_true', default=False, help="verbose")
args = parser.parse_args()
target_shape = tuple(args.shape)

print("target shape: ", target_shape)


def validate(filename):
    if args.verbose:
        print(filename)
    srcpath = os.path.join(args.src, filename)
    npz = np.load(srcpath)
    nparr = npz['arr_0']
    if args.verbose:
        print(nparr.shape)

    assert nparr.shape == target_shape


filelist = os.listdir(args.src)

if args.parallel:
    Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(
        delayed(validate)(filename) for filename in filelist)
else:
    for filename in filelist:
        validate(filename)
