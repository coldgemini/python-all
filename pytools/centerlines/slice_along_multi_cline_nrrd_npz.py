"""
slice multiple layers in nrrd stack into npz files
"""
import numpy as np
import os
from pypacks.ops3D.centerlines import get_center_lines
from pypacks.ops3D.centerlines import expand_along_curves
from pypacks.math.range_translation import scale_range
import nrrd
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src image dir")
parser.add_argument("--srcmask", type=str, help="src image dir")
parser.add_argument("--npz", type=str, help="dst npz slice dir")
parser.add_argument("--npzmask", type=str, help="dst npz mask slice dir")
parser.add_argument("--npzmaskvis", type=str, help="dst npz mask slice dir")
parser.add_argument("-c", "--precline", type=str, help="cline dir")
parser.add_argument("-w", "--window", type=int, default=50, help="cline dir")
parser.add_argument("--clist", type=str, help="cline list")
parser.add_argument("--slist", type=str, help="data list")
parser.add_argument("-n", "--n_jobs", type=int, default=2, help="parallel jobs")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
args = parser.parse_args()
srcdir = args.src
srcmask_dir = args.srcmask
npzdir = args.npz
npzmaskdir = args.npzmask
npzmaskvisdir = args.npzmaskvis
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
    for img_idx in range(20, c - 20):  # skip 20 layers both ends
        if not 1 in srcmsk_warp[:, :, img_idx]:
            continue
        slice_name = basename + '_cl_' + str(cl_idx) + '_slice_' + str(img_idx) + '.npz'
        slice_path = os.path.join(npzdir, slice_name)
        slice_mask_path = os.path.join(npzmaskdir, slice_name)
        slice_maskvis_path = os.path.join(npzmaskvisdir, slice_name)
        npz_slice = src_warp[:, :, img_idx - 20:img_idx + 20:5]  # TODO: modify second 20 to 21
        np.savez(slice_path, npz_slice)
        # print(np.unique(npz_slice))
        npzmask_slice = srcmsk_warp[:, :, img_idx]
        np.savez(slice_mask_path, npzmask_slice)
        npzmaskvis_slice = srcmsk_warp[:, :, img_idx] * 255
        np.savez(slice_maskvis_path, npzmaskvis_slice)


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
