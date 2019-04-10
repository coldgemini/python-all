import numpy as np
import os
import cv2
import argparse
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src image")
parser.add_argument("-d", "--dst", type=str, help="src image")
parser.add_argument("--sigma", type=float, default=4, help="src image")
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
if_vis = args.vis
n_jobs = args.n_jobs
parallel = args.parallel


# (cx, cy, M, m, theta) = ellipse_tuple


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def clipNstretch(image, clipLow, clipHigh, strhLow, strhHigh):
    img = image.astype(np.float64)
    img = np.clip(img, clipLow, clipHigh)
    img = (img - clipLow) / (clipHigh - clipLow)
    img = img * (strhHigh - strhLow)
    img = img + strhLow
    return img.astype(image.dtype)


def loss(pred, input_gt):
    inse = np.mean(pred[:, :] * input_gt[:, :])
    return -inse


# for filename in filelist:
def opt_ellipse(filename, ellipse_tuple):
    print(filename)
    basename = os.path.splitext(filename)[0]
    prob_basename = basename + '_prob.png'
    srcpath = os.path.join(srcdir, filename)
    dstpath = os.path.join(dstdir, filename)
    prob_dstpath = os.path.join(dstdir, prob_basename)
    # ellipse_init = ((128, 128), (40, 40), 10)  # initial mask ellipse
    (cx, cy, M, m, theta) = ellipse_tuple
    ellipse_init = ((cx, cy), (M, m), theta)  # initial mask ellipse
    image = cv2.imread(srcpath, 0)
    np_fix = cv2.resize(image, (256, 256))
    # np_mov = np.zeros_like(np_fix, dtype=np.uint8)
    mask = np.zeros_like(np_fix)
    # prob = np.zeros_like(np_fix, dtype=np.float32)
    cv2.ellipse(mask, ellipse_init, 255, -1)
    mask_idx = (mask != 255)
    # print(mask_idx)
    # mask = cv2.resize(mask, (512, 512))
    # cv2.imshow('mask', mask)
    # cv2.imshow('mask', mask_idx.astype(np.uint8) * 255)
    # cv2.moveWindow('mask', 1300, 100)
    np_fix_mask = np.ma.masked_array(np_fix, mask=mask_idx.astype(np.uint8))
    Iavg = np_fix_mask.mean()
    if if_vis:
        print("Iavg", Iavg)
    prob = np.where(True, gaussian(np_fix, Iavg, sigma), 0)
    prob_vis = clipNstretch(prob, 0, 1, 1, 254)
    prob_vis = prob_vis.astype(np.uint8)
    prob_train = clipNstretch(prob, 0, 1, -1, 1)
    # print('Iavg', Iavg)
    # print(np_fix_mask)
    if if_vis:
        cv2.imshow('fix', np_fix)
        cv2.moveWindow('fix', 100, 100)
        cv2.imshow('prob', prob_vis)
        cv2.moveWindow('prob', 700, 100)
    # key = cv2.waitKey(0)

    # trainable variables
    # cx = 128
    # cy = 128
    # M = 40
    # m = 40
    # theta = 40.0

    dx = 1
    dy = 1
    dM = 1
    dm = 1
    dtheta = 5.0

    ux = 100
    uy = 100
    uM = 100
    um = 100
    utheta = 500

    training_iters = steps

    for iter in range(training_iters):
        ellipse = ((cx, cy), (M, m), theta)
        np_mov = np.zeros_like(np_fix, dtype=np.int8)
        cv2.ellipse(np_mov, ellipse, 1, -1)
        np_mov = np_mov.astype(np.float32)
        np_mov = clipNstretch(np_mov, 0, 1, -1, 1)
        cost = loss(prob_train, np_mov)

        ellipse = ((cx + dx, cy), (M, m), theta)
        np_mov_d = np.zeros_like(np_fix, dtype=np.int8)
        cv2.ellipse(np_mov_d, ellipse, 1, -1)
        np_mov_d = np_mov_d.astype(np.float32)
        np_mov_d = clipNstretch(np_mov_d, 0, 1, -1, 1)
        cost_d = loss(prob_train, np_mov_d)
        err_dx = cost_d - cost

        ellipse = ((cx, cy + dy), (M, m), theta)
        np_mov_d = np.zeros_like(np_fix, dtype=np.int8)
        cv2.ellipse(np_mov_d, ellipse, 1, -1)
        np_mov_d = np_mov_d.astype(np.float32)
        np_mov_d = clipNstretch(np_mov_d, 0, 1, -1, 1)
        cost_d = loss(prob_train, np_mov_d)
        err_dy = cost_d - cost

        ellipse = ((cx, cy), (M + dM, m), theta)
        np_mov_d = np.zeros_like(np_fix, dtype=np.int8)
        cv2.ellipse(np_mov_d, ellipse, 1, -1)
        np_mov_d = np_mov_d.astype(np.float32)
        np_mov_d = clipNstretch(np_mov_d, 0, 1, -1, 1)
        cost_d = loss(prob_train, np_mov_d)
        err_dM = cost_d - cost

        ellipse = ((cx, cy), (M, m + dm), theta)
        np_mov_d = np.zeros_like(np_fix, dtype=np.int8)
        cv2.ellipse(np_mov_d, ellipse, 1, -1)
        np_mov_d = np_mov_d.astype(np.float32)
        np_mov_d = clipNstretch(np_mov_d, 0, 1, -1, 1)
        cost_d = loss(prob_train, np_mov_d)
        err_dm = cost_d - cost

        ellipse = ((cx, cy), (M, m), theta + dtheta)
        np_mov_d = np.zeros_like(np_fix, dtype=np.int8)
        cv2.ellipse(np_mov_d, ellipse, 1, -1)
        np_mov_d = np_mov_d.astype(np.float32)
        np_mov_d = clipNstretch(np_mov_d, 0, 1, -1, 1)
        cost_d = loss(prob_train, np_mov_d)
        err_dtheta = cost_d - cost

        error = err_dx + err_dy + err_dM + err_dm + err_dtheta

        # update ellipse parameters
        gx = -ux * err_dx
        gy = -uy * err_dy
        gM = -uM * err_dM
        gm = -um * err_dm
        gtheta = -utheta * err_dtheta

        cx += gx
        cy += gy
        M += gM
        m += gm
        theta += gtheta
        ellipse = ((cx, cy), (M, m), theta)

        if if_vis:
            print("Epoch", (iter + 1), ": error = ", "{0:.4f}".format(error),
                  ": gx = ", "{0:.4f}".format(gx),
                  ": gy = ", "{0:.4f}".format(gy),
                  ": gM = ", "{0:.4f}".format(gM),
                  ": gm = ", "{0:.4f}".format(gm),
                  ": gtheta = ", "{0:.4f}".format(gtheta))
            print("cx ", cx, ": cy ", cy, ": M ", M, ": m ", m, ": theta ", theta)

        # print("Training cost =", training_cost, ": tx = ", trans_X, ": ty= ", trans_Y)

        np_fix = np_fix.astype(np.uint8)
        np_fix_c = cv2.cvtColor(np_fix, cv2.COLOR_GRAY2BGR)
        prob_vis_c = cv2.cvtColor(prob_vis, cv2.COLOR_GRAY2BGR)
        np_mov = np.zeros_like(np_fix, dtype=np.uint8)
        cv2.ellipse(np_mov, ellipse, 255, -1)
        contours, _ = cv2.findContours(np_mov, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        cv2.drawContours(np_fix_c, [cnt], 0, (0, 255, 0), 1)
        cv2.drawContours(prob_vis_c, [cnt], 0, (0, 255, 0), 1)

        if if_vis:
            np_fix_c = cv2.resize(np_fix_c, (512, 512))
            prob_vis_c = cv2.resize(prob_vis_c, (512, 512))
            cv2.imshow('fix', np_fix_c)
            cv2.imshow('prob', prob_vis_c)
            # cv2.waitKey(0)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break

        cv2.imwrite(dstpath, np_fix_c)
        cv2.imwrite(prob_dstpath, prob_vis_c)


filelist = os.listdir(srcdir)

if parallel:
    Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(opt_ellipse)(filename, ellipse_tuple) for filename in filelist)
else:
    for filename in filelist:
        opt_ellipse(filename, ellipse_tuple)
