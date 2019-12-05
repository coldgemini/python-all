import os
import nibabel as nib
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src image dir")
parser.add_argument("-n", "--n_jobs", type=int, default=2, help="parallel jobs")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
parser.add_argument("-v", "--verbose", action='store_true', default=False, help="verbose")
args = parser.parse_args()


def show_shape(filename):
    print(filename)
    niftipath = os.path.join(args.src, filename)
    vol_file = nib.load(niftipath)
    data = vol_file.get_data()
    print(data.shape)


filelist = os.listdir(args.src)

if args.parallel:
    Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(
        delayed(show_shape)(filename) for filename in filelist)
else:
    for filename in filelist:
        show_shape(filename)
