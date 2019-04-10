import scipy.ndimage
import numpy as np


def get_centerlines_by_points(mask, point_cnt=500):
    #
    # # It is possible that there are multiple curves
    # print("mask")
    # print(mask.shape)
    # print(np.unique(mask))
    results = []
    x_np, y_np, z_np = np.where(mask == 1)
    curve = np.vstack((x_np, y_np, z_np))
    curve = np.transpose(curve)
    curve = curve[np.argsort(curve[:, 2])]
    # print("curve")
    # print(curve.shape)
    # print(curve)
    # for curve in curves:
    points = scipy.ndimage.zoom(curve.astype(float), (float(point_cnt) / len(curve), 1)).transpose()
    #     print(points[0])
    tck, u = scipy.interpolate.splprep(points, k=3)
    smoothed_points = scipy.interpolate.splev(u, tck, der=0)
    smoothed_points = np.array(smoothed_points).transpose()
    smoothed_points = smoothed_points
    results.append(smoothed_points)
    return results
