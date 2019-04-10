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


def opt_ellipse(image, ellipse_tuple, sigma=4, steps=500, if_vis=False):
    (cx, cy, M, m, theta) = ellipse_tuple
    ellipse_init = ((cx, cy), (M, m), theta)  # initial mask ellipse
    # np_fix = cv2.resize(image, (size, size))
    np_fix = image
    mask = np.zeros_like(np_fix)
    cv2.ellipse(mask, ellipse_init, 255, -1)
    mask_idx = (mask != 255)
    np_fix_mask = np.ma.masked_array(np_fix, mask=mask_idx.astype(np.uint8))
    Iavg = np_fix_mask.mean()
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

    (dx, dy, dM, dm, dtheta) = (1, 1, 1, 1, 5.0)  # tend to be bigger
    (ux, uy, uM, um, utheta) = (100, 100, 100, 100, 500)  # define learning rate

    for iter in range(steps):
        ellipse = ((cx, cy), (M, m), theta)
        np_mov = np.zeros_like(np_fix, dtype=np.int8)
        cv2.ellipse(np_mov, ellipse, 1, -1)
        np_mov = np_mov.astype(np.float32)
        np_mov = scale_range(np_mov, 0, 1, -1, 1)
        cost = loss(prob_train, np_mov)

        ellipse = ((cx + dx, cy), (M, m), theta)
        np_mov_d = np.zeros_like(np_fix, dtype=np.int8)
        cv2.ellipse(np_mov_d, ellipse, 1, -1)
        np_mov_d = np_mov_d.astype(np.float32)
        np_mov_d = scale_range(np_mov_d, 0, 1, -1, 1)
        cost_d = loss(prob_train, np_mov_d)
        err_dx = cost_d - cost

        ellipse = ((cx, cy + dy), (M, m), theta)
        np_mov_d = np.zeros_like(np_fix, dtype=np.int8)
        cv2.ellipse(np_mov_d, ellipse, 1, -1)
        np_mov_d = np_mov_d.astype(np.float32)
        np_mov_d = scale_range(np_mov_d, 0, 1, -1, 1)
        cost_d = loss(prob_train, np_mov_d)
        err_dy = cost_d - cost

        ellipse = ((cx, cy), (M + dM, m), theta)
        np_mov_d = np.zeros_like(np_fix, dtype=np.int8)
        cv2.ellipse(np_mov_d, ellipse, 1, -1)
        np_mov_d = np_mov_d.astype(np.float32)
        np_mov_d = scale_range(np_mov_d, 0, 1, -1, 1)
        cost_d = loss(prob_train, np_mov_d)
        err_dM = cost_d - cost

        ellipse = ((cx, cy), (M, m + dm), theta)
        np_mov_d = np.zeros_like(np_fix, dtype=np.int8)
        cv2.ellipse(np_mov_d, ellipse, 1, -1)
        np_mov_d = np_mov_d.astype(np.float32)
        np_mov_d = scale_range(np_mov_d, 0, 1, -1, 1)
        cost_d = loss(prob_train, np_mov_d)
        err_dm = cost_d - cost

        ellipse = ((cx, cy), (M, m), theta + dtheta)
        np_mov_d = np.zeros_like(np_fix, dtype=np.int8)
        cv2.ellipse(np_mov_d, ellipse, 1, -1)
        np_mov_d = np_mov_d.astype(np.float32)
        np_mov_d = scale_range(np_mov_d, 0, 1, -1, 1)
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

        if if_vis:
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

    return ellipse
