import scipy.ndimage
import skimage
import skimage.morphology
import numpy as np


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    except TypeError:
        return False
    return False


def norm(d): return d / np.sqrt(np.sum(d * d))


def sort_curve(mask, line_break_std=5, each_side_outlier=2, min_line_break_dist=0):
    """ Given a set of masked points evenly distributed on lines,
    partition them into ordered set such that each set represent a curve
    Return points in the same of [C, d] where C is the number of points"""
    # Get a ordered points by searching for the NN for each point
    coords = np.nonzero(mask.astype(int))
    coords = np.transpose(np.array(coords))
    for i in range(0, len(coords) - 1):
        dist = np.sum((coords[i + 1:] - coords[i]) ** 2, axis=1)
        dist1 = np.sum((coords[i + 1:] - coords[0]) ** 2, axis=1)
        idx = np.argmin(dist)
        idx1 = np.argmin(dist1)

        # Reverse the ordered part
        if dist[idx] > dist1[idx1]:
            coords[0:i + 1] = coords[0:i + 1][::-1]
            dist = dist1
            idx = idx1

        saved = coords[i + 1] + 0  # make a copy
        coords[i + 1] = coords[idx + i + 1]
        coords[idx + i + 1] = saved
    dist = np.sqrt(np.sum((coords[:-1] - coords[1:]) ** 2, axis=1))
    if each_side_outlier > 0:
        dist = np.sort(dist)[each_side_outlier:-each_side_outlier]
    dist_mean = np.mean(dist)
    dist_std = np.std(dist)

    def partition(coords):
        """Break coords into lines assuming that points are evenly distributed"""
        if len(coords) <= 2:
            return [coords]

        # Select the start point and reorder coords
        coords = np.concatenate([coords, [coords[0]]])
        dist = np.sqrt(np.sum((coords[:-1] - coords[1:]) ** 2, axis=1))
        idx = np.argmax(dist)

        coords = coords[:-1]
        coords = np.concatenate([coords[idx + 1:-1], coords[:idx + 1]])
        dist = np.sqrt(np.sum((coords[:-1] - coords[1:]) ** 2, axis=1))
        idx = np.argmax(dist)

        lines = [coords]
        if dist[idx] > dist_mean + line_break_std * dist_std + min_line_break_dist:
            lines = []
            lines.append(coords[:idx + 1])
            lines.append(coords[idx + 1:])

        # Recursively partition and reorder points
        if len(lines) > 1:
            new_lines = []
            for i in range(len(lines)):
                new_lines += partition(lines[i])
            lines = new_lines

        return lines

    lines = partition(coords)

    return lines


def resize_image3D(image3D, new_size, padding):
    """ enlarge image only """
    try:
        if is_number(new_size):
            new_size = [new_size] * 3
    except:
        pass

    if (image3D.shape[0] == new_size[0] and image3D.shape[1] == new_size[1] and image3D.shape[2] == new_size[2]):
        return image3D

    if (image3D.shape[0] > new_size[0]):
        start = (image3D.shape[0] - new_size[0]) / 2
        image3D = image3D[start:start + new_size[0]]
    if (image3D.shape[1] > new_size[1]):
        start = (image3D.shape[1] - new_size[1]) / 2
        image3D = image3D[:, start:start + new_size[1]]
    if (image3D.shape[2] > new_size[2]):
        start = (image3D.shape[2] - new_size[2]) / 2
        image3D = image3D[:, :, start:start + new_size[2]]

    new_img = np.zeros(new_size)
    new_img[:, :, :] = padding
    original_shape = image3D.shape

    offset = (new_size[0] - original_shape[0])
    left = np.round(offset / 2)
    right = new_size[0] - (offset - left)

    offset = (new_size[1] - original_shape[1])
    upper = np.round(offset / 2)
    lower = new_size[1] - (offset - upper)

    offset = (new_size[2] - original_shape[2])
    front = np.round(offset / 2)
    back = new_size[2] - (offset - front)

    new_img[int(left):int(right), int(upper):int(lower), int(front):int(back)] = image3D
    return new_img


def interpolate_cubic(image3D, xs, ys, zs):
    d = 1 - 1e-30
    xs0 = np.floor(xs).astype(int)
    xs1 = np.floor(xs + d).astype(int)
    ys0 = np.floor(ys).astype(int)
    ys1 = np.floor(ys + d).astype(int)
    zs0 = np.floor(zs).astype(int)
    zs1 = np.floor(zs + d).astype(int)
    xs0 = np.clip(xs0, 0, image3D.shape[0] - 1)
    xs1 = np.clip(xs1, 0, image3D.shape[0] - 1)
    ys0 = np.clip(ys0, 0, image3D.shape[1] - 1)
    ys1 = np.clip(ys1, 0, image3D.shape[1] - 1)
    zs0 = np.clip(zs0, 0, image3D.shape[2] - 1)
    zs1 = np.clip(zs1, 0, image3D.shape[2] - 1)
    values000 = image3D[xs0, ys0, zs0]
    values001 = image3D[xs0, ys0, zs1]
    values010 = image3D[xs0, ys1, zs0]
    values011 = image3D[xs0, ys1, zs1]
    values100 = image3D[xs1, ys0, zs0]
    values101 = image3D[xs1, ys0, zs1]
    values110 = image3D[xs1, ys1, zs0]
    values111 = image3D[xs1, ys1, zs1]

    values00 = values000 + (values001 - values000) * (zs - zs0)
    values01 = values010 + (values011 - values010) * (zs - zs0)
    values10 = values100 + (values101 - values100) * (zs - zs0)
    values11 = values110 + (values111 - values110) * (zs - zs0)

    values0 = values00 + (values01 - values00) * (ys - ys0)
    values1 = values10 + (values11 - values10) * (ys - ys0)

    values = values0 + (values1 - values0) * (xs - xs0)

    return values


def get_center_lines(mask, min_curve_len=10, skeleton=None, point_cnt=500):
    """ Based on skeletonize_3d
    Returned points are in the shape of [C, 3]"""
    if skeleton is None:
        skeleton = resize_image3D(mask.astype(float), np.array(mask.shape) + 10, 0)
        skeleton = scipy.ndimage.filters.gaussian_filter(skeleton, 3, truncate=5)
        skeleton = skeleton / np.max(skeleton)
        skeleton = skeleton > 0.2
        skeleton = skimage.morphology.skeletonize_3d(skeleton)
    if np.sum(skeleton) == 0:
        return []
    curves = sort_curve(skeleton, min_line_break_dist=5)
    curves = sorted(curves, key=lambda x: len(x))
    # print len(curves)

    # It is possible that there are multiple curves
    results = []
    for curve in curves:
        if len(curve) <= min_curve_len:
            continue

        points = scipy.ndimage.zoom(curve.astype(float), (float(point_cnt) / len(curve), 1)).transpose()

        tck, u = scipy.interpolate.splprep(points, k=3)
        smoothed_points = scipy.interpolate.splev(u, tck, der=0)
        smoothed_points = np.array(smoothed_points).transpose()
        smoothed_points = smoothed_points - 5
        results.append(smoothed_points)
    return results


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


def extract_patches_independent(image3D, points3D, direction3D, window):
    """
    Extract a 2D patch for each point.
    The patch is perpenticular to the specified directions.
    Make sure that the extracted window is in the boundary of image3D
    points3D is in shape [C, 3].

    The points are assumed to be independent
    """
    # print("inside extract")

    patches = []
    coords = []
    p2 = np.array([1, 0, 0])
    # for i in range(1):
    #     print("idx", i)
    for i in range(len(points3D)):
        p, d = points3D[i], direction3D[i]
        # print('p', p)
        # print('d', d)

        # Make sure that the coefficient for z is the greatest
        # per = np.argsort(np.abs(d))
        per = np.array([0, 1, 2])
        # print('per', per)
        p = p[per]
        d = d[per]
        # print('p', p)
        # print('d', d)

        # Select 3 points (denoted as P) in the voxel coord system
        p1 = norm(d)
        # p2 = norm(np.array([1, 0, - d[0] / (d[2] + 1e-10)]))
        p2 = norm(p2 - np.matmul(p1, p2) * p1)
        # p3 = norm(np.array([0, 1, - d[1] / (d[2] + 1e-10)]))
        p3 = np.cross(p1, p2)
        P = np.array([p1, p2, p3]).transpose()
        # print('p1', p1)
        # print('p2', p2)
        # print('p3', p3)
        # print('P', P)

        # Select 3 points (denoted as Q) in the simple coord system
        Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).transpose()

        #         print P
        #         print Q

        # Calculate rotation matrix T such that
        # P = T * Q
        # T = P * Q'
        T = np.matmul(P, np.linalg.inv(Q))

        ys, zs = np.mgrid[-window:window + 1, -window:window + 1]
        xs = np.zeros((2 * window + 1) ** 2)

        P = np.matmul(np.array(T), np.array([xs.flatten(), ys.flatten(), zs.flatten()]))
        P += p.reshape(3, 1)

        # Reorder x,y,z
        P = P[np.argsort(per)]

        xs, ys, zs = P
        patch = interpolate_cubic(image3D, xs, ys, zs)

        patch = patch.reshape(window * 2 + 1, window * 2 + 1)
        xs = xs.reshape(window * 2 + 1, window * 2 + 1)
        ys = ys.reshape(window * 2 + 1, window * 2 + 1)
        zs = zs.reshape(window * 2 + 1, window * 2 + 1)

        patches.append(patch)
        coords.append([xs, ys, zs])

    patches = np.array(patches)
    coords = np.array(coords)
    return patches, coords


def expand_along_curves(image3D, center_lines, win):
    """ Example usage:
        center_lines = get_center_lines(mask3D, point_cnt=500)
        results = expand_along_curves(image3D, center_lines, win)
     """
    results = []
    for cline in center_lines:
        points = []
        for i in range(len(cline)):
            diff = cline[i].astype(int) - cline[i - 1].astype(int)
            if np.sum(diff * diff) == 0:
                continue
            points.append(cline[i])
        points = np.array(points)
        patches, coords = extract_patches_independent(image3D, points[:-1], points[1:] - points[:-1], win)
        results.append([patches, coords])
    return results


def restore_from_fields(empty_mask, coords, full_mask):
    """
    transposes coords array
    :param empty_mask:
    :param coords:
    :param full_mask:
    :return:
    """
    coords = np.transpose(coords, (2, 3, 0, 1))
    mh, mw, mc = empty_mask.shape
    h, w, c = full_mask.shape
    for y in range(h):
        for x in range(w):
            for z in range(c):
                # print(x, y, z)
                label = full_mask[x, y, z]
                idx = coords[x, y, z]
                idx = idx.astype(np.int32)
                # print(idx)
                if not (0 <= idx[0] < mh and 0 <= idx[1] < mw and 0 <= idx[2] < mc):
                    continue
                idx = idx.astype(np.uint16)
                # print(idx)
                empty_mask[idx[0], idx[1], idx[2]] = label
    return
