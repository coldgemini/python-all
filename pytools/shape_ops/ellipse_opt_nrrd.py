import numpy as np
import os
import cv2
import nrrd
from pypacks.optimization.ellipse import opt_ellipse
from pypacks.math.range_translation import scale_range
import argparse
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src image dir")
parser.add_argument("-d", "--dst", type=str, help="src image dir")
parser.add_argument("--sigma", type=float, default=4, help="src image")
parser.add_argument("--dilate", type=float, default=9, help="src image")
parser.add_argument('--ellipse', nargs='+', default=(128, 128, 40, 40, 40), type=int)
parser.add_argument("-i", "--iter", type=int, default=1000, help="iter steps")
parser.add_argument("-v", "--vis", action='store_true', default=False, help="if visualize")
parser.add_argument("-n", "--n_jobs", type=int, default=40, help="parallel jobs")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
args = parser.parse_args()
srcdir = args.src
dstdir = args.dst
steps = args.iter
ellipse_tuple = tuple(args.ellipse)
sigma = args.sigma
dilate = args.dilate
if_vis = args.vis
n_jobs = args.n_jobs
parallel = args.parallel


# for filename in filelist:
def draw_ellipse(filename, ellipse_tuple):
    print(filename)
    srcpath = os.path.join(srcdir, filename)
    dstpath = os.path.join(dstdir, filename)
    npsrc, header = nrrd.read(srcpath)
    yS, xS, zS = npsrc.shape
    # print(npsrc.shape)
    mask = np.zeros_like(npsrc, dtype=np.uint8)
    npsrc = npsrc.astype(np.float16)
    npsrc_2d = np.average(npsrc, axis=2)
    npsrc_2d = npsrc_2d.astype(np.int32)
    npsrc_2d = scale_range(npsrc_2d, -1000, 1000, 0, 255)
    npsrc_2d = npsrc_2d.astype(np.uint8)

    image = npsrc_2d
    np_fix = cv2.resize(image, (256, 256))
    mask_2d = np.zeros_like(np_fix)
    ellipse = opt_ellipse(np_fix, ellipse_tuple, sigma=sigma, steps=steps)

    ((cx, cy), (M, m), theta) = ellipse
    (M, m) = (M + dilate, m + dilate)
    ellipse = ((cx, cy), (M, m), theta)

    np_cnt_help = np.zeros_like(np_fix, dtype=np.uint8)
    cv2.ellipse(np_cnt_help, ellipse, 255, -1)
    contours, _ = cv2.findContours(np_cnt_help, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    cv2.drawContours(mask_2d, [cnt], 0, 1, -1)
    mask_2d = cv2.resize(mask_2d, (xS, yS), cv2.INTER_NEAREST)
    mask_2d = np.expand_dims(mask_2d, -1)
    mask = mask + mask_2d  # broadcast
    # print(mask.shape)

    nrrd.write(dstpath, mask)


filelist = os.listdir(srcdir)

if parallel:
    Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(draw_ellipse)(filename, ellipse_tuple) for filename in filelist)
else:
    for filename in filelist:
        draw_ellipse(filename, ellipse_tuple)
