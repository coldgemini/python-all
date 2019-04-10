import numpy as np
import os
# from util import get_center_lines
# from util import test_get_center_lines
from pypacks.ops3D.centerlines import get_centerlines_by_points
from util import expand_along_curves
import nrrd
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src image")
parser.add_argument("--srcmsk", type=str, help="mask image")
parser.add_argument("-d", "--dst", type=str, help="dst image")
parser.add_argument("--dstmsk", type=str, help="mask image")
parser.add_argument("-c", "--precline", type=str, help="cline dir")
parser.add_argument("-l", "--list", type=str, help="processing list")
parser.add_argument("-n", "--n_jobs", type=int, default=40, help="parallel jobs")
parser.add_argument("-g", "--logdir", type=str, default='logdir', help="log directory")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
args = parser.parse_args()
srcdir = args.src
dstdir = args.dst
srcmskdir = args.srcmsk
dstmskdir = args.dstmsk
preclinedir = args.precline
listfile = args.list
n_jobs = args.n_jobs
logdir = args.logdir
parallel = args.parallel

# read listfile
listfile_h = open(listfile, "r")
file_lines = listfile_h.readlines()
file_lines = [line.rstrip() for line in file_lines]
listfile_h.close()


# for filename in file_lines:
def crop_along_cline(filename):
    print(filename)
    # logpath = os.path.join(logdir, filename + ".txt")
    # file = open(logpath, 'w')
    # file.write(repr(filename) + '\n')

    srcpath = os.path.join(srcdir, filename)
    dstpath = os.path.join(dstdir, filename)
    clinepath = os.path.join(preclinedir, filename)
    if srcmskdir:
        srcmskpath = os.path.join(srcmskdir, filename)
        dstmskpath = os.path.join(dstmskdir, filename)
    # print(clinepath)

    srcdata, header = nrrd.read(srcpath)
    clinedata, header = nrrd.read(clinepath)
    if srcmskdir:
        srcmskdata, header = nrrd.read(srcmskpath)
    # print("srcmsk", np.unique(srcmskdata))
    # file.write("srcmsk: " + repr(np.unique(srcmskdata)) + '\n')
    # print(srcdata.shape)
    # print(srcmskdata.shape)
    # print(clinedata.shape)

    # clines = get_center_lines(clinedata, point_cnt=500)
    clines = get_centerlines_by_points(clinedata, point_cnt=500)

    # print(type(clines))
    # print(len(clines))

    src_results = expand_along_curves(srcdata, clines, 50)
    if srcmskdir:
        srcmsk_results = expand_along_curves(srcmskdata, clines, 50)

    # print(len(src_results))
    # print(len(srcmsk_results))

    src_rst = src_results[0][0]
    src_rst = src_rst.astype(np.int32)
    src_rst = np.transpose(src_rst, (1, 2, 0))
    nrrd.write(dstpath, src_rst)
    if srcmskdir:
        msk_rst = srcmsk_results[0][0]
        # print(type(msk_rst))
        msk_rst = msk_rst.astype(np.int16)
        # print(type(msk_rst))
        msk_rst = np.transpose(msk_rst, (1, 2, 0))
        # print("dstmsk", np.unique(msk_rst))
        nrrd.write(dstmskpath, msk_rst)
    # print("finished", filename)

    # file.close()
    return


if parallel:
    Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(crop_along_cline)(filename) for filename in file_lines)
else:
    for filename in file_lines:
        crop_along_cline(filename)
