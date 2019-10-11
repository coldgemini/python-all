import os
import nrrd
import argparse
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--srcdir", type=str, help="src image folder")
parser.add_argument("-d", "--dstdir", type=str, help="dst image folder")
parser.add_argument("-i", "--index", type=int, default=0, help="dst image folder")
parser.add_argument("-n", "--n_jobs", type=int, default=30, help="parallel jobs")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
args = parser.parse_args()
srcdir = args.srcdir
dstdir = args.dstdir
index = args.index
n_jobs = args.n_jobs
parallel = args.parallel


def slice_along_z(filename):
    print(filename)
    srcpath = os.path.join(srcdir, filename)
    dstpath = os.path.join(dstdir, filename)

    npsrc, header = nrrd.read(srcpath)
    # npdst = npsrc[:, :, index:index + 3]
    # npdst = npsrc[:, :, 0::10]
    npdst = npsrc[:, :, 50:100:2]
    nrrd.write(dstpath, npdst)


filelist = os.listdir(srcdir)

if parallel:
    Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(slice_along_z)(filename) for filename in filelist)
else:
    for filename in filelist:
        slice_along_z(filename)
