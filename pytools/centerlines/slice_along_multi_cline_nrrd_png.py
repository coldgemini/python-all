import numpy as np
import os
from pypacks.ops3D.centerlines import get_center_lines
from pypacks.ops3D.centerlines import expand_along_curves
from pypacks.math.range_translation import scale_range
import nrrd
from PIL import Image
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src image dir")
parser.add_argument("--srcmask", type=str, help="src image dir")
parser.add_argument("--png", type=str, help="dst image dir")
parser.add_argument("--pngmask", type=str, help="dst image dir")
parser.add_argument("--pngmaskvis", type=str, help="dst image dir")
parser.add_argument("-c", "--precline", type=str, help="cline dir")
parser.add_argument("-w", "--window", type=int, default=80, help="cline dir")
parser.add_argument("--clist", type=str, help="cline list")
parser.add_argument("--slist", type=str, help="data list")
parser.add_argument("-n", "--n_jobs", type=int, default=3, help="parallel jobs")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
args = parser.parse_args()
srcdir = args.src
srcmask_dir = args.srcmask
pngdir = args.png
pngmask_dir = args.pngmask
pngmaskvis_dir = args.pngmaskvis
preclinedir = args.precline
win = args.window
clistfile = args.clist
slistfile = args.slist
n_jobs = args.n_jobs
parallel = args.parallel


def crop_along_cline(sfilename, cl_idx):
    print("data:", sfilename)
    basename = os.path.splitext(sfilename)[0]
    srcpath = os.path.join(srcdir, sfilename)
    srcmask_path = os.path.join(srcmask_dir, sfilename)

    srcdata, _ = nrrd.read(srcpath)
    srcmsk_data, _ = nrrd.read(srcmask_path)
    # print("srcmask: ", np.unique(srcmsk_data))

    clines = get_center_lines(clinedata, point_cnt=500)
    src_results = expand_along_curves(srcdata, clines, win)
    srcmsk_results = expand_along_curves(srcmsk_data, clines, win)

    src_rst = src_results[0][0].astype(np.int32)
    # print("src: ", np.unique(src_rst))
    src_rst = scale_range(src_rst, -1000, 1000, 1, 254)
    src_rst = src_rst.astype(np.uint8)
    src_warp = np.transpose(src_rst, (1, 2, 0))
    # print("src: ", np.unique(src_warp))

    srcmsk_rst = srcmsk_results[0][0]
    # print("srcmask: ", np.unique(srcmsk_rst))
    srcmsk_rst = (srcmsk_rst >= 0.5)
    srcmsk_rst = srcmsk_rst.astype(np.uint8)
    srcmsk_warp = np.transpose(srcmsk_rst, (1, 2, 0))
    # print("srcmask: ", np.unique(srcmsk_warp))

    (h, w, c) = src_rst.shape
    for img_idx in range(c):
        if not 1 in srcmsk_warp[:, :, img_idx]:
            continue
        imgname = basename + '_cl_' + str(cl_idx) + '_slice_' + str(img_idx) + '.png'
        imgpath = os.path.join(pngdir, imgname)
        imgmaskpath = os.path.join(pngmask_dir, imgname)
        imgmaskvispath = os.path.join(pngmaskvis_dir, imgname)
        im = Image.fromarray(src_warp[:, :, img_idx], mode="L")
        im.save(imgpath)
        im = Image.fromarray(srcmsk_warp[:, :, img_idx], mode="L")
        im.save(imgmaskpath)
        im = Image.fromarray(255 * srcmsk_warp[:, :, img_idx], mode="L")
        im.save(imgmaskvispath)


slistfile_h = open(slistfile, "r")
sfile_lines = slistfile_h.readlines()
sfile_lines = [line.rstrip() for line in sfile_lines]
slistfile_h.close()

clistfile_h = open(clistfile, "r")
cfile_lines = clistfile_h.readlines()
cfile_lines = [line.rstrip() for line in cfile_lines]
clistfile_h.close()

for cl_idx in range(len(cfile_lines)):
    cline_basename = cfile_lines[cl_idx]
    print("cline: ", cline_basename)
    cline_path = os.path.join(preclinedir, cline_basename)
    clinedata, header = nrrd.read(cline_path)

    if parallel:
        Parallel(n_jobs=n_jobs, backend="multiprocessing")(
            delayed(crop_along_cline)(sfilename, cl_idx) for sfilename in sfile_lines)
    else:
        for sfilename in sfile_lines:
            crop_along_cline(sfilename, cl_idx)
