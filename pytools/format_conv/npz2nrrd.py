import os
import sys
import numpy as np
import nrrd
# from glob import glob
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--srcdir", type=str, help="src image folder")
parser.add_argument("-d", "--dstdir", type=str, help="dst image folder")
parser.add_argument("-n", "--n_jobs", type=int, default=3, help="parallel jobs")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
args = parser.parse_args()

filelist = os.listdir(args.srcdir)


# for name in files:
def conv_npz_nrrd(filename):
    print(filename)
    sys.stdout.flush()
    npzpath = os.path.join(args.srcdir, filename)
    filebase = os.path.splitext(filename)[0]
    nrrdbasename = filebase + '.nrrd'
    nrrdpath = os.path.join(args.dstdir, nrrdbasename)

    # print("npz", npzpath)
    # print("nrrd", nrrdpath)

    npz = np.load(npzpath)
    # print(npz.files)
    nparr = npz['arr_0']
    # print(nparr.shape)
    nrrd.write(nrrdpath, nparr)


# print(cfile_lines)
if args.parallel:
    Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(
        delayed(conv_npz_nrrd)(filename) for filename in filelist)
else:
    for filename in filelist:
        conv_npz_nrrd(filename)
