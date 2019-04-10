import numpy as np
import cv2
from pypacks.math.range_translation import scale_range
from pypacks.math.functions import gaussian


def loss(pred, input_gt):
    inse = np.mean(pred[:, :] * input_gt[:, :])
    return -inse


def get_cost(prob_train, ellipse):
    np_mov = np.zeros_like(prob_train, dtype=np.int8)
    cv2.ellipse(np_mov, ellipse, 1, -1)
    np_mov = np_mov.astype(np.float32)
    np_mov = scale_range(np_mov, 0, 1, -1, 1)
    cost = loss(prob_train, np_mov)
    return cost


def generate_ellipses(ellipse, delta_tuple):
    delta_list = list(delta_tuple)
    ((cx, cy), (M, m), theta) = ellipse
    ellipse_as_list = [cx, cy, M, m, theta]
    ellipses = []
    for idx in range(len(delta_tuple)):
        new_ellipse_list = ellipse_as_list.copy()
        new_ellipse_list[idx] += delta_list[idx]
        new_ellipse_tuple = (
            (new_ellipse_list[0], new_ellipse_list[1]), (new_ellipse_list[2], new_ellipse_list[3]), new_ellipse_list[4])
        ellipses.append(new_ellipse_tuple)
    return ellipses


def opt_ellipse(image, ellipse_tuple, sigma=4, steps=500, if_vis=False):
    (cx, cy, M, m, theta) = ellipse_tuple
    ellipse_init = ((cx, cy), (M, m), theta)  # initial mask ellipse

    # initialize probability image
    np_fix = image
    mask = np.zeros_like(np_fix)
    cv2.ellipse(mask, ellipse_init, 255, -1)
    mask_idx = (mask != 255)
    np_fix_masked = np.ma.masked_array(np_fix, mask=mask_idx.astype(np.uint8))
    Iavg = np_fix_masked.mean()
    prob = np.where(True, gaussian(np_fix, Iavg, sigma), 0)
    prob_train = scale_range(prob, 0, 1, -1, 1)

    if if_vis:
        prob_vis = scale_range(prob, 0, 1, 1, 254)
        prob_vis = prob_vis.astype(np.uint8)
        cv2.imshow('fix', np_fix)
        cv2.moveWindow('fix', 100, 100)
        cv2.imshow('prob', prob_vis)
        cv2.moveWindow('prob', 700, 100)
        cv2.waitKey(0)

    (dx, dy, dM, dm, dtheta) = (1, 1, 1, 1, 5.0)  # tends to be bigger
    (ux, uy, uM, um, utheta) = (100, 100, 100, 100, 500)  # define learning rate

    for iter in range(steps):

        # calc errors in each direction
        ellipse = ((cx, cy), (M, m), theta)
        cost = get_cost(prob_train, ellipse)
        ell_x, ell_y, ell_M, ell_m, ell_theta = generate_ellipses(ellipse, (dx, dy, dM, dm, dtheta))
        (err_dx, err_dy, err_dM, err_dm, err_dtheta) = map(lambda ellipse: get_cost(prob_train, ellipse) - cost,
                                                           [ell_x, ell_y, ell_M, ell_m, ell_theta])
        error = err_dx + err_dy + err_dM + err_dm + err_dtheta

        # update ellipse parameters
        (gx, gy, gM, gm, gtheta) = (-ux * err_dx, -uy * err_dy, -uM * err_dM, -um * err_dm, -utheta * err_dtheta)
        (cx, cy, M, m, theta) = map(lambda x, y: x + y, [cx, cy, M, m, theta], [gx, gy, gM, gm, gtheta])

        if if_vis:
            print("Epoch", (iter + 1), ": error = ", "{0:.4f}".format(error),
                  ": gx = ", "{0:.4f}".format(gx),
                  ": gy = ", "{0:.4f}".format(gy),
                  ": gM = ", "{0:.4f}".format(gM),
                  ": gm = ", "{0:.4f}".format(gm),
                  ": gtheta = ", "{0:.4f}".format(gtheta))
            print("cx ", cx, ": cy ", cy, ": M ", M, ": m ", m, ": theta ", theta)

        if if_vis:
            ellipse = ((cx, cy), (M, m), theta)
            np_fix = np_fix.astype(np.uint8)
            np_fix_c = cv2.cvtColor(np_fix, cv2.COLOR_GRAY2BGR)
            prob_vis_c = cv2.cvtColor(prob_vis, cv2.COLOR_GRAY2BGR)
            np_mov = np.zeros_like(np_fix, dtype=np.uint8)
            cv2.ellipse(np_mov, ellipse, 255, -1)
            contours, _ = cv2.findContours(np_mov, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            cv2.drawContours(np_fix_c, [cnt], 0, (0, 255, 0), 1)
            cv2.drawContours(prob_vis_c, [cnt], 0, (0, 255, 0), 1)
            np_fix_c = cv2.resize(np_fix_c, (512, 512))
            prob_vis_c = cv2.resize(prob_vis_c, (512, 512))
            cv2.imshow('fix', np_fix_c)
            cv2.imshow('prob', prob_vis_c)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break

    ellipse = ((cx, cy), (M, m), theta)
    return ellipse
