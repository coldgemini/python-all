import numpy as np
import scipy.ndimage
import os
import nrrd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src image dir")
parser.add_argument("-d", "--dst", type=str, help="dst image dir")
parser.add_argument("-l", "--list", type=str, help="cline list")
args = parser.parse_args()
srcdir = args.src
dstdir = args.dst
listfile = args.list

# print(clistfile)
listfile_h = open(listfile, "r")
file_lines = listfile_h.readlines()
file_lines = [line.rstrip() for line in file_lines]
listfile_h.close()

point_cnt = 500
for filename in file_lines:
    print(filename)
    srcpath = os.path.join(srcdir, filename)
    dstpath = os.path.join(dstdir, filename)
    mask, header = nrrd.read(srcpath)
    mask = mask.astype(np.uint8)

    x_np, y_np, z_np = np.where(mask == 1)
    curve = np.vstack((x_np, y_np, z_np))
    curve = np.transpose(curve)
    curve = curve[np.argsort(curve[:, 2])]
    # print("curve")
    # print(curve.shape)
    # print(curve)
    # for curve in curves:

    # linear interpolation
    points = scipy.ndimage.zoom(curve.astype(float), (float(point_cnt) / len(curve), 1)).transpose()
    print(points.shape)
    # points = np.transpose(points)
    dim, length = points.shape
    for idx in range(length):
        locX, locY, locZ = points[0][idx], points[1][idx], points[2][idx]
        locX, locY, locZ = int(locX), int(locY), int(locZ)
        mask[locX, locY, locZ] = 1

    # spline interpolation
    tck, u = scipy.interpolate.splprep(points, k=3)
    smoothed_points = scipy.interpolate.splev(u, tck, der=0)
    smoothed_points = np.array(smoothed_points).transpose()
    smoothed_points = smoothed_points
    print(smoothed_points.shape)

    length, dim = smoothed_points.shape
    for idx in range(length):
        locX, locY, locZ = smoothed_points[idx][0], smoothed_points[idx][1], smoothed_points[idx][2]
        locX, locY, locZ = int(locX), int(locY), int(locZ)
        mask[locX, locY, locZ] = 2
        print(locX, locY, locZ)

    nrrd.write(dstpath, mask)
