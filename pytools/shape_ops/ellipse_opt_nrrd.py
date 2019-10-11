import numpy as np
import os
import cv2
import nrrd
from scipy.ndimage import zoom
from pypacks.debug.ellipse import opt_ellipse, opt_ellipse_3d
from pypacks.math.range_translation import scale_range
import argparse
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src image dir")
parser.add_argument("-d", "--dst", type=str, help="src image dir")
parser.add_argument("--size", type=int, default=256, help="src image dir")
parser.add_argument("--sigma", type=float, default=4, help="src image")
parser.add_argument("--dilate", type=float, default=9, help="src image")
parser.add_argument('--ellipse', nargs='+', default=(128, 128, 40, 40, 40), type=int)
parser.add_argument("--steps", type=int, default=1000, help="iter steps")
parser.add_argument("-v", "--vis", action='store_true', default=False, help="if visualize")
parser.add_argument("-n", "--n_jobs", type=int, default=40, help="parallel jobs")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
args = parser.parse_args()
srcdir = args.src
dstdir = args.dst
size = args.size
steps = args.steps
ellipse_tuple = tuple(args.ellipse)
sigma = args.sigma
dilate = args.dilate
if_vis = args.vis
n_jobs = args.n_jobs
parallel = args.parallel
sigma_3d = sigma / 2.0


# for filename in filelist:
def draw_ellipse(filename, ellipse_tuple):
    print(filename)
    srcpath = os.path.join(srcdir, filename)
    dstpath = os.path.join(dstdir, filename)
    npsrc, header = nrrd.read(srcpath)
    yS, xS, zS = npsrc.shape
    # print(npsrc.shape)
    npsrc = npsrc.astype(np.float16)

    # initialize by 2d ellipse
    npsrc_2d = np.average(npsrc, axis=2)
    npsrc_2d = npsrc_2d.astype(np.int32)
    npsrc_2d = scale_range(npsrc_2d, -1000, 1000, 0, 255)
    npsrc_2d = npsrc_2d.astype(np.uint8)

    image = npsrc_2d
    np_fix = cv2.resize(image, (256, 256))
    ellipse_2d = opt_ellipse(np_fix, ellipse_tuple, sigma=sigma, steps=steps * 3)
    ((cx, cy), (M, m), theta) = ellipse_2d
    ellipse_tuple = (cx, cy, M, m, theta)
    print("finished 2d initialization!")
    print(ellipse_2d)

    npsrc = scale_range(npsrc, -1000, 1000, 0, 255)
    npsrc = npsrc.astype(np.uint8)

    # image_3d = npsrc
    # np_fix = cv2.resize(image_3d, (256, 256))
    np_fix = zoom(npsrc, (size / xS, size / yS, 1), order=0, mode='nearest')
    mask_3d = np.zeros_like(np_fix, dtype=np.uint8)
    # print("mask_3d initial shape: ", mask_3d.shape)
    # print("mask_3d initial dtype: ", mask_3d.dtype)
    # print("np_fix shape: ", np_fix.shape)
    # mask_2d = np.zeros_like(np_fix, dtype=np.uint8)
    ellipse_3d = opt_ellipse_3d(np_fix, ellipse_tuple, sigma=sigma_3d, steps=steps, if_vis=True)
    # dilate_3d = np.full((zS), dilate)
    ellipse_3d[:, 2], ellipse_3d[:, 3] = ellipse_3d[:, 2] + dilate, ellipse_3d[:, 3] + dilate

    for idx in range(zS):
        # print("iter: {}".format(idx))
        ellipse = (
            (ellipse_3d[idx][0], ellipse_3d[idx][1]), (ellipse_3d[idx][2], ellipse_3d[idx][3]), ellipse_3d[idx][4])
        # print(ellipse)
        np_cnt_help = np.zeros_like(np_fix[:, :, 0], dtype=np.uint8)
        cv2.ellipse(np_cnt_help, ellipse_2d, 2, -1)
        cv2.ellipse(np_cnt_help, ellipse, 1, -1)
        mask_3d[:, :, idx] = np_cnt_help

    # cv2.imshow('mask_3d', mask_3d[:, :, idx])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print("mask_3d unique: ", np.unique(mask_3d))
    # print(mask_3d.shape)
    # mask_2d = cv2.resize(mask_2d, (xS, yS), cv2.INTER_NEAREST)
    mask_3d = zoom(mask_3d, (xS / size, yS / size, 1), order=0, mode='nearest')
    # print("mask_3d final shape: ", mask_3d.shape)
    # mask_2d = np.expand_dims(mask_2d, -1)
    # mask = mask + mask_2d  # broadcast
    # print(mask.shape)

    nrrd.write(dstpath, mask_3d)


filelist = os.listdir(srcdir)

if parallel:
    Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(draw_ellipse)(filename, ellipse_tuple) for filename in filelist)
else:
    for filename in filelist:
        draw_ellipse(filename, ellipse_tuple)
