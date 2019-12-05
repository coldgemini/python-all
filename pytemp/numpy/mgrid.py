import numpy as np


# ys, zs = np.mgrid[0:3, 0:3]
#
# print(ys)
# print(zs)
#
# a = np.eye(3)
# print(a)
# # a[ys, zs] = np.max(abs(ys - 1), abs(zs - 1))
# b = np.maximum(abs(ys - 1), abs(zs - 1))
# print(b)


def get_weight(mat_shape):
    xS, yS, zS = mat_shape
    xM, yM, zM = xS // 2, yS // 2, zS // 2
    print(xM, yM, zM)
    xs, ys, zs = np.mgrid[0:xS, 0:yS, 0:zS]
    print(xs)
    print(ys)
    print(zs)
    # dis = np.maximum(abs(xs - xM), abs(ys - yM), abs(zs - zM))
    stack = np.stack((abs(xs - xM), abs(ys - yM), abs(zs - zM)), axis=-1)
    dis = np.max(stack, axis=-1)
    print(dis.shape)
    print(f"dis: {dis}")
    Mdis = np.max(dis)
    print(f"Mdis: {Mdis}")
    dis_inv = Mdis + 1 - dis
    print(f"dis_inv: {dis_inv}")
    Mdis_inv = np.max(dis_inv)
    print(f"Mdis_inv: {Mdis_inv}")
    dis_inv_norm = dis_inv / Mdis_inv
    return dis_inv_norm


if __name__ == '__main__':
    w = get_weight((5, 5, 5))
    print(w)
