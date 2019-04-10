import numpy as np
import os
import cv2
from pypacks.optimization.ellipse import opt_ellipse
import argparse
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src image")
parser.add_argument("-d", "--dst", type=str, help="src image")
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
    image = cv2.imread(srcpath, 0)
    np_fix = cv2.resize(image, (256, 256))
    ellipse = opt_ellipse(np_fix, ellipse_tuple, sigma=sigma, steps=steps)
    ((cx, cy), (M, m), theta) = ellipse
    (M, m) = (M + dilate, m + dilate)
    ellipse = ((cx, cy), (M, m), theta)

    np_cnt_help = np.zeros_like(np_fix, dtype=np.uint8)
    np_fix_c = cv2.cvtColor(np_fix, cv2.COLOR_GRAY2BGR)
    cv2.ellipse(np_cnt_help, ellipse, 255, -1)
    contours, _ = cv2.findContours(np_cnt_help, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    cv2.drawContours(np_fix_c, [cnt], 0, (0, 255, 0), 1)

    cv2.imwrite(dstpath, np_fix_c)


filelist = os.listdir(srcdir)

if parallel:
    Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(draw_ellipse)(filename, ellipse_tuple) for filename in filelist)
else:
    for filename in filelist:
        draw_ellipse(filename, ellipse_tuple)
