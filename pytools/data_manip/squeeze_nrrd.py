import os
import numpy as np
import nrrd
import argparse
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--srcdir", type=str, help="src image folder")
parser.add_argument("-d", "--dstdir", type=str, help="dst image folder")
parser.add_argument("-n", "--n_jobs", type=int, default=30, help="parallel jobs")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
args = parser.parse_args()
srcdir = args.srcdir
dstdir = args.dstdir
n_jobs = args.n_jobs
parallel = args.parallel


def squeeze_along_z(filename):
    print(filename)
    srcpath = os.path.join(srcdir, filename)
    dstpath = os.path.join(dstdir, filename)

    npsrc, header = nrrd.read(srcpath)
    npsrc = npsrc.astype(np.float16)
    npdst = np.average(npsrc, axis=2)
    npdst = npdst.astype(np.int32)
    nrrd.write(dstpath, npdst)


filelist = os.listdir(srcdir)

if parallel:
    Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(squeeze_along_z)(filename) for filename in filelist)
else:
    for filename in filelist:
        squeeze_along_z(filename)
