"""
crop nrrd by single continous center line mask for testing
"""
import numpy as np
import os
from pypacks.ops3D.centerlines import get_center_lines
from pypacks.ops3D.centerlines import expand_along_curves
import nrrd
import sys
from joblib import Parallel, delayed
# from progress.bar import Bar
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src image dir")
parser.add_argument("-d", "--dst", type=str, help="dst image dir")
parser.add_argument("-c", "--precline", type=str, help="cline dir")
parser.add_argument("-f", "--field", type=str, help="field dir")
parser.add_argument("--clist", type=str, help="cline list")
parser.add_argument("--slist", type=str, help="data list")
parser.add_argument("-n", "--n_jobs", type=int, default=3, help="parallel jobs")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
parser.add_argument("-m", "--if_mask", action='store_true', default=False, help="if extracting mask")
parser.add_argument("-v", "--verbose", action='store_true', default=False, help="verbose")
parser.add_argument("--progress", action='store_true', default=False, help="progress bar")
args = parser.parse_args()
srcdir = args.src
dstdir = args.dst
preclinedir = args.precline
field_dir = args.field
clistfile = args.clist
slistfile = args.slist
n_jobs = args.n_jobs
parallel = args.parallel
if_mask = args.if_mask
verbose = args.verbose
# progress = args.progress


def crop_along_cline(sfilename):
    if verbose:
        print("data", sfilename)
    srcpath = os.path.join(srcdir, sfilename)
    dstpath = os.path.join(dstdir, sfilename)
    if field_dir:
        field_path = os.path.join(field_dir, sfilename)

    if not os.path.isfile(srcpath):
        print("cannot find", srcpath)
        return
    srcdata, header = nrrd.read(srcpath)
    # if verbose:
    #     print("original mask: ", np.unique(srcdata))
    srcdata = srcdata.astype(np.int32)

    # clines = get_center_lines(clinedata, point_cnt=500)
    src_results = expand_along_curves(srcdata, clines, 50)

    if if_mask:
        src_rst = src_results[0][0]
        # if verbose:
        #     print("src_rst dtype: ", src_rst.dtype)
        #     print("transformed mask0: ", np.unique(src_rst))
        src_rst = src_rst.astype(np.uint8)
    else:
        src_rst = src_results[0][0].astype(np.int32)

    src_rst = np.transpose(src_rst, (1, 2, 0))
    if field_dir:
        src_field = src_results[0][1]
    # if verbose:
    #     print("transformed mask: ", np.unique(src_rst))
    nrrd.write(dstpath, src_rst)
    if field_dir:
        nrrd.write(field_path, src_field)

    # if progress:
    #     bar.next()


clistfile_h = open(clistfile, "r")
cfile_lines = clistfile_h.readlines()
cfile_lines = [line.rstrip() for line in cfile_lines]
clistfile_h.close()
# print(cfile_lines[0])
cline_basename = cfile_lines[0]
cline_path = os.path.join(preclinedir, cline_basename)
if not os.path.isfile(cline_path):
    print("cannot find", cline_path)
    sys.exit(-1)
clinedata, header = nrrd.read(cline_path)
clines = get_center_lines(clinedata, point_cnt=500)

slistfile_h = open(slistfile, "r")
sfile_lines = slistfile_h.readlines()
sfile_lines = [line.rstrip() for line in sfile_lines]
slistfile_h.close()

# if progress:
#     bar = Bar('Processing', max=len(sfile_lines))

if parallel:
    # Parallel(n_jobs=n_jobs, backend="multiprocessing", require='sharedmem')(
    #     delayed(crop_along_cline)(sfilename) for sfilename in sfile_lines)
    Parallel(n_jobs=n_jobs, require='sharedmem')(
        delayed(crop_along_cline)(sfilename) for sfilename in sfile_lines)
else:
    for sfilename in sfile_lines:
        crop_along_cline(sfilename)

# if progress:
#     bar.finish()
