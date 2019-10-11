import numpy as np
import cv2
from pypacks.math.range_translation import scale_range
from pypacks.math.functions import gaussian


def loss(input_gt_prob, pred):
    inse = np.mean(pred[:, :] * input_gt_prob[:, :])
    return -inse


def loss_3d(input_gt_prob, pred, np_init, f_g=0.5, f_c=0.3, f_c2=0.3, f_i=0.6):
    inse = np.mean(pred[:, :, :] * input_gt_prob[:, :, :], axis=(0, 1))
    pred_up = np.roll(pred, 1, axis=2)
    pred_up[:, :, 0] = 0
    inse_up = np.mean(pred[:, :, :] * pred_up[:, :, :], axis=(0, 1))
    pred_dn = np.roll(pred, -1, axis=2)
    pred_dn[:, :, -1] = 0
    inse_dn = np.mean(pred[:, :, :] * pred_dn[:, :, :], axis=(0, 1))

    inse_updn = np.mean(pred_up[:, :, :] * pred_dn[:, :, :], axis=(0, 1))

    inse_init = np.mean(pred[:, :, :] * np_init[:, :, :], axis=(0, 1))
    # print("inse: {}, up: {}, dn: {}".format(inse, inse_up, inse_dn))
    return -(
            f_g * inse + f_i * inse_init ** 2 + f_c * inse_up ** 2 + f_c * inse_dn ** 2 + f_c2 * inse_updn ** 2)


def get_cost(prob_train, ellipse):
    np_mov = np.zeros_like(prob_train, dtype=np.uint8)
    cv2.ellipse(np_mov, ellipse, 1, -1)
    np_mov = np_mov.astype(np.float32)
    np_mov = scale_range(np_mov, 0, 1, -1, 1)
    cost = loss(prob_train, np_mov)
    return cost


def get_cost_3d(prob_train_3d, ellipse_tuple, ellipse_3d):
    ellipse_init = ((ellipse_tuple[0], ellipse_tuple[0]), (ellipse_tuple[0], ellipse_tuple[0]), ellipse_tuple[0])
    np_init = np.zeros_like(prob_train_3d, dtype=np.uint8)
    np_helper = np.zeros_like(np_init[:, :, 0])
    cv2.ellipse(np_helper, ellipse_init, 1, -1)
    np_helper = np.expand_dims(np_helper, axis=-1)
    np_init = np_init + np_helper  # broadcast

    np_mov = np.zeros_like(prob_train_3d, dtype=np.uint8)
    for idx in range(len(ellipse_3d)):
        cx, cy, M, m, theta = ellipse_3d[idx].tolist()
        ellipse = ((cx, cy), (M, m), theta)
        np_helper = np.zeros_like(np_mov[:, :, 0])
        cv2.ellipse(np_helper, ellipse, 1, -1)
        np_mov[:, :, idx] = np_helper
    np_mov = np_mov.astype(np.float32)
    np_mov = scale_range(np_mov, 0, 1, -1, 1)
    cost_3d = loss_3d(prob_train_3d, np_mov, np_init)
    return cost_3d


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


def generate_ellipses_3d(ellipse_3d, delta_tuple):
    # print(ellipse_3d.dtype)
    delta_list = list(delta_tuple)
    # delta_list_3d = np.zeros_like(ellipse_3d)
    # delta_list_3d[:, 0], delta_list_3d[:, 1], delta_list_3d[:, 2], delta_list_3d[:, 3], delta_list_3d[:, 4] = map(
    #     lambda x, y: x + y,
    #     [delta_list_3d[:, 0], delta_list_3d[:, 1], delta_list_3d[:, 2], delta_list_3d[:, 3], delta_list_3d[:, 4]],
    #     delta_list)
    ellipses_3d = []
    for idx in range(len(delta_tuple)):
        new_ellipse_3d = ellipse_3d.copy()
        new_ellipse_3d[:, idx] += delta_list[idx]  # broadcast
        ellipses_3d.append(new_ellipse_3d)
    return ellipses_3d


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


def opt_ellipse_3d(image_3d, ellipse_tuple, sigma=4, steps=500, if_vis=False):
    # initialize probability image
    np_fix_3d = image_3d
    yS, xS, zS = image_3d.shape
    mask_3d = np.zeros_like(np_fix_3d)
    prob_train_3d = np.zeros_like(mask_3d, dtype=np.float64)

    (cx, cy, M, m, theta) = ellipse_tuple
    (cx, cy, M, m, theta) = map(lambda x: np.full((zS), float(x)), [cx, cy, M, m, theta])
    ellipse_init = np.stack((cx, cy, M, m, theta), axis=-1)
    # print(ellipse_init.shape)
    ellipse_3d = ellipse_init

    for idx in range(zS):
        ellipse = (
            (ellipse_init[idx][0], ellipse_init[idx][1]), (ellipse_init[idx][2], ellipse_init[idx][3]),
            ellipse_init[idx][4])
        mask = np.zeros((yS, xS), dtype=np.uint8)
        cv2.ellipse(mask, ellipse, 255, -1)
        mask_idx = (mask != 255)
        np_fix_masked = np.ma.masked_array(np_fix_3d[:, :, idx], mask=mask_idx.astype(np.uint8))
        Iavg = np_fix_masked.mean()
        prob = np.where(True, gaussian(np_fix_3d[:, :, idx], Iavg, sigma), 0)
        # print("prob.dtype: ", prob.dtype)
        prob_train_3d[:, :, idx] = scale_range(prob, 0, 1, -1, 1)

    if if_vis:
        for idx in range(0, zS, 10):
            prob_train_3d_slice = scale_range(prob_train_3d[:, :, idx], -1, 0, 1, 254).astype(np.uint8)
            cv2.imshow('prob', prob_train_3d_slice)
            cv2.waitKey(0)

    (dx, dy, dM, dm, dtheta) = (1.0, 1.0, 1.0, 1.0, 5.0)  # tends to be bigger
    # (ux, uy, uM, um, utheta) = (100, 100, 100, 100, 500)  # define learning rate
    (ux, uy, uM, um, utheta) = (200, 200, 200, 200, 800)  # define learning rate

    for iter in range(steps):
        print("iter step {}".format(iter))
        # calc errors in each direction
        cost_3d = get_cost_3d(prob_train_3d, ellipse_tuple, ellipse_3d)
        ell_x, ell_y, ell_M, ell_m, ell_theta = generate_ellipses_3d(ellipse_3d, (dx, dy, dM, dm, dtheta))
        # (err_dx, err_dy, err_dM, err_dm, err_dtheta) = map(
        #     lambda ellipse_3d: get_cost_3d(prob_train_3d, ellipse_3d) - cost_3d,
        #     [ell_x, ell_y, ell_M, ell_m, ell_theta])
        cost_dx = get_cost_3d(prob_train_3d, ellipse_tuple, ell_x)
        err_dx = cost_dx - cost_3d
        cost_dy = get_cost_3d(prob_train_3d, ellipse_tuple, ell_y)
        err_dy = cost_dy - cost_3d
        cost_dM = get_cost_3d(prob_train_3d, ellipse_tuple, ell_M)
        err_dM = cost_dM - cost_3d
        cost_dm = get_cost_3d(prob_train_3d, ellipse_tuple, ell_m)
        err_dm = cost_dm - cost_3d
        cost_dtheta = get_cost_3d(prob_train_3d, ellipse_tuple, ell_theta)
        err_dtheta = cost_dtheta - cost_3d

        # update ellipse parameters
        (gx, gy, gM, gm, gtheta) = (-ux * err_dx, -uy * err_dy, -uM * err_dM, -um * err_dm, -utheta * err_dtheta)
        update_3d = np.stack((gx, gy, gM, gm, gtheta), axis=-1)
        ellipse_3d += update_3d

        debug_idx = 10
        if if_vis:
            print("Epoch", iter),
            print(": ellipse_x = ", (ell_x[debug_idx]),
                  ": ellipse_y = ", (ell_y[debug_idx]),
                  ": ellipse_M = ", (ell_M[debug_idx]),
                  ": ellipse_m = ", (ell_m[debug_idx]),
                  ": ellipse_theta = ", (ell_theta[debug_idx]))

            print("Cost 3D = {0:.4f}".format(cost_3d[debug_idx]))
            print(": cost_x = ", "{0:.4f}".format(cost_dx[debug_idx]),
                  ": cost_y = ", "{0:.4f}".format(cost_dy[debug_idx]),
                  ": cost_M = ", "{0:.4f}".format(cost_dM[debug_idx]),
                  ": cost_m = ", "{0:.4f}".format(cost_dm[debug_idx]),
                  ": theta = ", "{0:.4f}".format(cost_dtheta[debug_idx]))
            print(": ex = ", "{0:.4f}".format(err_dx[debug_idx]),
                  ": ey = ", "{0:.4f}".format(err_dy[debug_idx]),
                  ": eM = ", "{0:.4f}".format(err_dM[debug_idx]),
                  ": em = ", "{0:.4f}".format(err_dm[debug_idx]),
                  ": etheta = ", "{0:.4f}".format(err_dtheta[debug_idx]))
            print(": gx = ", "{0:.4f}".format(gx[debug_idx]),
                  ": gy = ", "{0:.4f}".format(gy[debug_idx]),
                  ": gM = ", "{0:.4f}".format(gM[debug_idx]),
                  ": gm = ", "{0:.4f}".format(gm[debug_idx]),
                  ": gtheta = ", "{0:.4f}".format(gtheta[debug_idx]))
            print("cx :{:.4f}, cy :{:.4f}, M :{:.4f}, m :{:.4f}, theta :{:.4f}".format(ellipse_3d[debug_idx][0],
                                                                                       ellipse_3d[debug_idx][1],
                                                                                       ellipse_3d[debug_idx][2],
                                                                                       ellipse_3d[debug_idx][3],
                                                                                       ellipse_3d[debug_idx][4]))
            # print(ellipse_3d[debug_idx])

    return ellipse_3d
