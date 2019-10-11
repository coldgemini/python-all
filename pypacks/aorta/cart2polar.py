import numpy as np
import cv2
from scipy.ndimage.interpolation import geometric_transform
from functools import partial


def transform2polar(coords, shape, center):
    """
    :param coords: coords in output image
    :return: coords in input image
    """
    theta = np.pi * coords[1] / shape[1]  # only 180 deg
    radius = coords[0] - shape[0] / 2
    i = center[1] + radius * np.sin(theta)
    j = center[0] + radius * np.cos(theta)
    return i, j


def transform2cart(coords, shape, center):
    """
    :param coords: coords in output image
    :return: coords in input image
    """
    xindex, yindex = coords
    x = xindex - center[0]
    y = yindex - center[1]
    r = np.sqrt(x ** 2.0 + y ** 2.0)
    theta = np.arctan2(y, x, where=True)
    if theta < 0:
        theta = theta % np.pi  # only take one 180 deg range
        r = -r
    theta_index = theta * shape[1] / np.pi
    r = r + shape[0] / 2
    return (r, theta_index)


def topolar(img, center, order=5):
    # max_radius = 0.5 * np.linalg.norm(img.shape)

    # def transform(coords):
    #     """
    #     :param coords: coords in output image
    #     :return: coords in input image
    #     """
    #     theta = 2.0 * np.pi * coords[1] / (img.shape[1] - 1.)
    #     radius = max_radius * coords[0] / img.shape[0]
    #     i = center[0] - radius * np.sin(theta)
    #     j = center[1] - radius * np.cos(theta)
    #     return i, j
    transform2polar_p = partial(transform2polar, center=center, shape=img.shape)

    polar = geometric_transform(img, transform2polar_p, order=order, mode='nearest', prefilter=True)

    return polar


def tocart(img, center, order=5):
    # max_radius = 0.5 * np.linalg.norm(img.shape)

    # def transform(coords):
    #     """
    #     :param coords: coords in output image
    #     :return: coords in input image
    #     """
    #     xindex, yindex = coords
    #     x = xindex - center[1]
    #     y = yindex - center[0]
    #     r = np.sqrt(x ** 2.0 + y ** 2.0) * (img.shape[1] / max_radius)
    #     theta = np.arctan2(y, x, where=True)
    #     theta_index = (theta + np.pi) * img.shape[1] / (2 * np.pi)
    #     return (r, theta_index)
    transform2cart_p = partial(transform2cart, center=center, shape=img.shape)

    polar = geometric_transform(img, transform2cart_p, order=order, mode='nearest', prefilter=True)

    return polar


if __name__ == '__main__':
    img = np.zeros((256, 256), np.uint8)
    cv2.ellipse(img, (128, 128), (64, 64), 45, startAngle=0, endAngle=360, color=255, thickness=-1)
    print(img.dtype)
    pol = topolar(img, (128, 128))
    print(pol.dtype)
    # pol = topolar(img, (100, 100))
    # res = tocart(pol)
    res = tocart(pol, (128, 128))
    print(res.dtype)
    # pol1 = pol / 255
    # res = res / 255
    cv2.imshow('original', img)
    cv2.moveWindow('original', 0, 0)
    cv2.imshow('linearpolar2', pol)
    cv2.moveWindow('linearpolar2', 300, 0)
    cv2.imshow('linearpolar3', res)
    cv2.moveWindow('linearpolar3', 600, 0)
    cv2.waitKey(0)
