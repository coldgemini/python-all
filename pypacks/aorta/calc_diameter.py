import numpy as np
import cv2
from functools import partial
from pypacks.aorta.cart2polar import topolar, transform2polar


def calc_diameters(mask, verbose=False):
    """
    :param mask: input image mask
    :return: two point pairs, long and short diameter ends
    """
    if not 1 in np.unique(mask):  ## empty mask
        if verbose:
            print("empty mask!!")
        return [(0, 0), ((0, 0), (0, 0)), ((0, 0), (0, 0))]

    assert list(np.unique(mask)) == [0, 1], "input mask contain only [0, 1]"

    ret, thresh = cv2.threshold(mask, 0.5, 1, 0)

    M = cv2.moments(thresh)

    # calculate x,y coordinate of center
    cY = int(M["m10"] / M["m00"])
    cX = int(M["m01"] / M["m00"])

    pol = topolar(thresh, (cY, cX))
    sum = np.sum(pol, axis=0)
    long_idx = np.argmax(sum)
    short_idx = np.argmin(sum)
    if verbose:
        print("long_idx: ", long_idx)
        print("short_idx: ", short_idx)

    transform2polar_p = partial(transform2polar, center=(cY, cX), shape=mask.shape)

    long_col = pol[:, long_idx]
    if verbose:
        print("long_col")
        print(long_col)
    mask_in_long_col = np.nonzero(long_col == 1)
    if len(mask_in_long_col[0]) == 0:
        if verbose:
            print("empty long column !!")
        return [(cY, cX), ((0, 0), (0, 0)), ((0, 0), (0, 0))]

    if verbose:
        print("mask_in_long_col")
        print(mask_in_long_col)
    long_coords_polar = ((mask_in_long_col[0][0], long_idx), (mask_in_long_col[0][-1], long_idx))
    long_coords_cart = list(map(transform2polar_p, long_coords_polar))
    long_coords_cart = [tuple(map(int, elem)) for elem in long_coords_cart]
    long_coords_cart = [(elem[1], elem[0]) for elem in long_coords_cart]

    short_col = pol[:, short_idx]
    if verbose:
        print("short_col")
        print(short_col)
    mask_in_short_col = np.nonzero(short_col == 1)
    if len(mask_in_short_col[0]) == 0:
        if verbose:
            print("empty short column !!")
        return [(cY, cX), ((0, 0), (0, 0)), ((0, 0), (0, 0))]

    if verbose:
        print("mask_in_short_col")
        print(mask_in_short_col)
        print(len(mask_in_short_col[0]))
    short_coords_polar = ((mask_in_short_col[0][0], short_idx), (mask_in_short_col[0][-1], short_idx))
    short_coords_cart = list(map(transform2polar_p, short_coords_polar))
    short_coords_cart = [tuple(map(int, elem)) for elem in short_coords_cart]
    short_coords_cart = [(elem[1], elem[0]) for elem in short_coords_cart]

    return [(cY, cX), long_coords_cart, short_coords_cart]


def main():
    mask = np.zeros((256, 256), np.uint8)
    cv2.ellipse(mask, (100, 100), (100, 20), 45, startAngle=0, endAngle=360, color=1, thickness=-1)

    center, long_dia_ends, short_dia_ends = calc_diameters(mask)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_bgr = mask_bgr * 255

    pt1, pt2 = long_dia_ends
    cv2.line(mask_bgr, pt1, pt2, [0, 255, 0], thickness=3)
    pt1, pt2 = short_dia_ends
    cv2.line(mask_bgr, pt1, pt2, [255, 0, 0], thickness=3)

    cv2.line(mask_bgr, center, center, [0, 0, 255], thickness=3)

    cv2.imshow('ellipse', mask_bgr)
    cv2.moveWindow('ellipse', 300, 300)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
