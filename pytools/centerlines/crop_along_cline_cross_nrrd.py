"""
crop nrrd by different continous center line masks for data augmentation
"""
import numpy as np
import os
from pypacks.ops3D.centerlines import get_center_lines
from pypacks.ops3D.centerlines import expand_along_curves
import nrrd
from joblib import Parallel, delayed
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
args = parser.parse_args()
srcdir = args.src
dstdir = args.dst
preclinedir = args.precline
field_dir = args.field
clistfile = args.clist
slistfile = args.slist
n_jobs = args.n_jobs
parallel = args.parallel


def clipNstretch(image, clipLow, clipHigh, strhLow, strhHigh):
    img = image.astype(np.float64)
    img = np.clip(img, clipLow, clipHigh)
    img = (img - clipLow) / (clipHigh - clipLow)
    img = img * (strhHigh - strhLow)
    img = img + strhLow
    return img.astype(image.dtype)


clistfile_h = open(clistfile, "r")
cfile_lines = clistfile_h.readlines()
cfile_lines = [line.rstrip() for line in cfile_lines]
clistfile_h.close()
print(cfile_lines[0])
# sys.exit(0)

slistfile_h = open(slistfile, "r")
sfile_lines = slistfile_h.readlines()
sfile_lines = [line.rstrip() for line in sfile_lines]
slistfile_h.close()

pproclist = []
for clidx, filename in enumerate(cfile_lines):
    pproclist.append([clidx, filename, sfile_lines])


def crop_along_cline(pprocItem):
    clidx = pprocItem[0]
    cfilename = pprocItem[1]
    sfile_lines = pprocItem[2]
    print("cline", cfilename)
    clinepath = os.path.join(preclinedir, cfilename)
    clinedata, header = nrrd.read(clinepath)
    for sfilename in sfile_lines:
        print("data", sfilename)
        basename = os.path.splitext(sfilename)[0]
        dfilename = basename + '_cl' + str(clidx) + '.nrrd'
        srcpath = os.path.join(srcdir, sfilename)
        dstpath = os.path.join(dstdir, dfilename)
        field_path = os.path.join(field_dir, dfilename)

        srcdata, header = nrrd.read(srcpath)

        clines = get_center_lines(clinedata, point_cnt=500)
        src_results = expand_along_curves(srcdata, clines, 50)

        src_rst = src_results[0][0].astype(np.int32)
        src_rst = np.transpose(src_rst, (1, 2, 0))
        nrrd.write(dstpath, src_rst)


if parallel:
    print("parallel!!!")
    Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(crop_along_cline)(pprocItem) for pprocItem in pproclist)
else:
    for pprocItem in pproclist:
        crop_along_cline(pprocItem)
