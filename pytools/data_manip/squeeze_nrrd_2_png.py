import os
import numpy as np
import nrrd
from PIL import Image
import argparse
from pypacks.math.range_translation import scale_range
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
    basename = os.path.splitext(filename)[0]
    srcpath = os.path.join(srcdir, filename)
    dstpath = os.path.join(dstdir, basename + '.png')

    npsrc, header = nrrd.read(srcpath)
    npsrc = npsrc.astype(np.float16)
    npdst = np.average(npsrc, axis=2)
    npdst = npdst.astype(np.int32)
    npdst = scale_range(npdst, -1000, 1000, 0, 255)
    npdst = npdst.astype(np.uint8)
    img_dst = Image.fromarray(npdst)
    img_dst.save(dstpath)


filelist = os.listdir(srcdir)

if parallel:
    Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(squeeze_along_z)(filename) for filename in filelist)
else:
    for filename in filelist:
        squeeze_along_z(filename)
