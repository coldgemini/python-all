"""
This script
"""

# import numpy as np  # linear algebra
import skimage, os
# from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, \
#     reconstruction, binary_closing
# from skimage.morphology import convex_hull_image
# from skimage.measure import label, regionprops, perimeter
# from skimage.morphology import binary_dilation, binary_opening
# from skimage.filters import roberts, sobel
from skimage import measure, feature
# from skimage.segmentation import clear_border
# from skimage import data
from scipy import ndimage
import scipy
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.misc
from skimage import data, transform
# import numpy as np
# import util
# import sys
import random
from scipy.spatial import ConvexHull
# from scipy import interpolate
import skimage.draw
import math
import scipy.ndimage
# import simpleTexture3D
# from scipy import ndimage as ndi
# import networkx as nx
# from networkx import *
import numpy as np
from numpy import linalg as LA

'''
This function is used to convert the world coordinates to voxel coordinates using 
the origin and spacing of the ct_scan
'''


def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return voxel_coordinates


'''
This function is used to convert the voxel coordinates to world coordinates using 
the origin and spacing of the ct_scan.
'''


def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates


def seq(start, stop, step=1):
    n = int(round((stop - start) / float(step)))
    if n > 1:
        return ([start + step * i for i in range(n + 1)])
    else:
        return ([])


'''
This function is used to create spherical regions in binary masks
at the given locations and radius. spacing is a single number which
is applied to all 3 dimentions. 
'''


def draw_balls(image3D_shape, cands, origin, spacing):
    # make empty matrix, which will be filled with the mask
    image_mask = np.zeros(image3D_shape)

    # run over all the nodules in the lungs
    for ca in cands:
        # get middel x-,y-, and z-worldcoordinate of the nodule
        radius = np.ceil(ca[4]) / 2
        coord_x = ca[1]
        coord_y = ca[2]
        coord_z = ca[3]
        image_coord = np.array((coord_x, coord_y, coord_z))

        # determine voxel coordinate given the worldcoordinate
        image_coord = world_2_voxel(image_coord, origin, spacing)

        # determine the range of the nodule
        noduleRange = seq(-radius, radius, spacing)

        # create the mask
        for x in noduleRange:
            for y in noduleRange:
                for z in noduleRange:
                    coords = world_2_voxel(np.array((coord_x + x, coord_y + y, coord_z + z)), origin, spacing)
                    if (np.linalg.norm(image_coord - coords) * spacing) < radius:
                        image_mask[int(coords[0]), int(coords[1]), int(coords[2])] = int(1)

    return image_mask


'''
This function is used to translate candidate coordiantes 
'''


def world_2_voxel_batch(cands, origin, spacing):
    # run over all the nodules in the lungs
    coords = []
    for ca in cands:
        # determine voxel coordinate given the worldcoordinate
        image_coord = world_2_voxel(ca, origin, spacing)
        image_coord = [int(each) for each in image_coord]
        coords.append(image_coord)
    coords = np.array(coords)
    return coords


'''
This function is used to get patches of 2D image. 
cands is 3D coordinates in world coordinate
'''


def get_2D_patches(image3D, cands, origin, spacing, win_size):
    patches = []
    for ca in cands:
        # determine voxel coordinate given the worldcoordinate
        coords = world_2_voxel(ca, origin, spacing)
        x, y, z = [int(each) for each in coords]
        xlow = max(x - win_size, 0)
        xhigh = min(x + win_size, image3D.shape[0])
        ylow = max(y - win_size, 0)
        yhigh = min(y + win_size, image3D.shape[1])

        patches.append(image3D[xlow:xhigh, ylow:yhigh, z])
    return patches


"""
Randomly rotate a image
"""


def random_rotate(image3D):
    theta = np.deg2rad(10)
    tx = 0
    ty = 0

    S, C = np.sin(theta), np.cos(theta)

    # Rotation matrix, angle theta, translation tx, ty
    H = np.array([[C, -S, tx],
                  [S, C, ty],
                  [0, 0, 1]])

    # Translation matrix to shift the image center to the origin
    r, c = img.shape
    T = np.array([[1, 0, -c / 2.],
                  [0, 1, -r / 2.],
                  [0, 0, 1]])

    # Skew, for perspective
    S = np.array([[1, 0, 0],
                  [0, 1.3, 0],
                  [0, 1e-3, 1]])

    img_rot = transform.homography(img, H)
    img_rot_center_skew = transform.homography(img, S.dot(np.linalg.inv(T).dot(H).dot(T)))
    return image3D


def normalizeDouble(image, MIN_BOUND=-1000.0, MAX_BOUND=400.0, grandglass=False):
    # Use two different cutoffs to do normalization
    image = image.reshape(list(image.shape) + [1])

    image1 = normalize(image)
    image2 = normalize(image, grandglass=True)

    image = np.concatenate([image1, image2], axis=len(image.shape) - 1)
    return image


def normalize(image, MIN_BOUND=-1000.0, MAX_BOUND=400.0, grandglass=False, bone=False):
    # The HU bound to filter lung
    if grandglass:
        MIN_BOUND = -1000.0
        MAX_BOUND = -500.0
    if bone:
        MIN_BOUND = -450.0
        MAX_BOUND = 1050.0
    image = (image - MIN_BOUND) / (float(MAX_BOUND) - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def unnormalize(image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
    # The HU bound to filter lung
    image = image * (MAX_BOUND - MIN_BOUND) - 1000
    return image


def contourize(image, threshold=range(-700, 101, 100), has_raw=True):
    if has_raw:
        mask = np.zeros(image.shape + (len(threshold) + 1,))
    else:
        mask = np.zeros(image.shape + (len(threshold),))
    for i, th in enumerate(threshold):
        try:
            verts, faces = measure.marching_cubes(image, th)
            verts = verts.astype(int)
            mask[verts[:, 0], verts[:, 1], verts[:, 2], i] = 1
        except ValueError:  # , e:
            pass
        except IndexError:  # , e:  # marching_cubes may raise this exception if th is not in the range of HU in image
            pass
    if has_raw:
        mask[:, :, :, -1] = normalize(image)
    return mask


def batch_contourize(image, threshold=range(-700, 101, 100), has_raw=True):
    masks = []
    for each in image:
        masks.append(contourize(each, threshold, has_raw))
    masks = np.array(masks)
    return masks


def seg_lung(image):
    """ Assume input image is in x,y,z format """
    std = 3
    orig_image = image
    image = scipy.ndimage.filters.gaussian_filter(orig_image, (std, std, 0.0001))
    mask = (image > -500)
    bodymask = np.zeros(mask.shape)
    for i in range(image.shape[2]):
        # Find the body mask
        curr_mask = mask[:, :, i]
        labels = measure.label(curr_mask, background=0)
        vals, counts = np.unique(labels, return_counts=True)
        ## Do not count background
        vc = sorted(zip(vals, counts), reverse=True, key=lambda x: x[1] * (x[0] > 0))
        if len(vc) <= 1: continue
        curr_mask = (labels == vc[0][0])

        # It is possible that the body is weakly connected to the bed
        # Find the lowest point of the bodymask and erose by 5 pixel to penetrate the bed
        # Also assume that 5 pixel will not touch the lung
        # The penetration will connect the interior region of the bed to the corner
        slicex, slicey = scipy.ndimage.find_objects(curr_mask)[0]
        curr_mask[:, slicey.stop - 5:slicey.stop] = 0

        # Find the convex hull of top half of the body to avoid leaked lungs
        # However, this step may introduce false positive lung regions around
        # the boundary. The false positves will be removed later.
        curr_mask = curr_mask.astype(int)
        boundary = curr_mask - scipy.ndimage.binary_erosion(curr_mask, iterations=1)
        points = np.nonzero(boundary.astype(int))
        points = np.transpose(np.array(points))
        # Take points above only to avoid the bed
        c = scipy.ndimage.measurements.center_of_mass(curr_mask)
        hull = ConvexHull(points)
        hv = hull.vertices
        # Append the first element to the last to close the curve
        hv = np.concatenate([hv, [hv[0]]], axis=0)
        hull = np.zeros(boundary.shape)
        for j in range(len(hv) - 1):
            # If one of the point is below the center of mass, skip
            if points[hv[j], 1] >= c[1] or points[hv[j + 1], 1] >= c[1]: continue
            # If two ponts are far away, skip
            # This happens if the image contains two arms
            dist = math.sqrt(np.sum((points[hv[j]] - points[hv[j + 1]]) ** 2))
            if dist > 200: continue
            rr, cc, val = skimage.draw.line_aa(points[hv[j], 0], points[hv[j], 1], points[hv[j + 1], 0],
                                               points[hv[j + 1], 1])
            hull[rr, cc] = (val > 0)
        curr_mask = np.maximum(curr_mask, hull)
        bodymask[:, :, i] = curr_mask

        # Find the lung mask by removing the bodymask and the connected components
        # that connect to the 4 corners
        labels = measure.label(1 - curr_mask, background=0)
        for corner in np.unique([labels[0, 0], labels[-1, 0], labels[0, -1], labels[-1, -1]]):
            curr_mask = np.maximum(curr_mask, labels == corner)
        curr_mask = 1 - curr_mask

        # Enlarge the hull such that any region with center hitting the enlarged hull will be removed.
        # Dilation 20 times would remove majority of false positves. It also possible
        # a women with huge breast may leave the region untouched. In this case, the
        # top connected components in 3D may solve the problem
        hull = scipy.ndimage.binary_dilation(hull, iterations=20)

        # For each connected component of the remained regions, remove regions
        # with centers hitting the enlarged hull
        labels = measure.label(curr_mask, background=0)
        vals, counts = np.unique(labels, return_counts=True)
        vc = sorted(zip(vals, counts), reverse=True, key=lambda x: x[1] * (x[0] > 0))
        for j in range(min(10, len(vc) - 1)):  # Do not remove the whole background
            region = labels == vc[j][0]
            center = np.array(scipy.ndimage.measurements.center_of_mass(region)).astype(int)
            if hull[center[0], center[1]] > 0:
                curr_mask -= region

        mask[:, :, i] = curr_mask

    # Gaussisan smoothing may lead to smaller lung region, recover such regions
    mask = scipy.ndimage.binary_dilation(mask, iterations=1)
    mask = mask * (orig_image < -300)
    mask = scipy.ndimage.binary_erosion(mask, iterations=1)

    if np.sum(mask) == 0:
        raise Exception("Lung segmentation error, the size of the lung is 0")

    mask = get_top_components(mask, volume_ratio=0.1)
    return mask, bodymask


def extract_lung_region(image3D, mask3D, image_size=None):
    """ This function will be used by NoduleMask2D. It is also used by generate_2D to generate training data """
    xstart, xstop, ystart, ystop = image3D.shape[0], 0, image3D.shape[1], 0
    for slicex, slicey, slicez in scipy.ndimage.find_objects(mask3D.astype(int)):
        xstart = min(xstart, slicex.start)
        ystart = min(ystart, slicey.start)
        xstop = max(xstop, slicex.stop)
        ystop = max(ystop, slicey.stop)
    image3D = image3D[xstart: xstop, ystart: ystop]
    mask3D = mask3D[xstart: xstop, ystart: ystop]

    lung_shape = np.array(image3D.shape)

    ratio = 1
    if image_size is not None:
        model_shape = np.array((image_size[0], image_size[1], image3D.shape[2]))
        ratio = lung_shape / model_shape.astype(float)
        new_shape = (model_shape[0], model_shape[1], image3D.shape[2])
        image3D = util.zoom_3D_cuda(image3D, new_shape=new_shape)
        mask3D = util.zoom_3D_cuda(mask3D, new_shape=new_shape)
        mask3D = (mask3D > 0.8).astype(int)

    return image3D, mask3D, ratio, (xstart, ystart), lung_shape


def get_top_components(mask, top_cnt=None, volume_ratio=0, min_vol=0):
    # Find the top largest connected components
    labels = measure.label(mask, background=0)
    vals, counts = np.unique(labels, return_counts=True)
    vc = sorted(zip(vals, counts), reverse=True, key=lambda x: x[1] * (x[0] > 0))
    vc = [each for each in vc if each[0] > 0]
    total_size = np.sum(mask)
    mask = np.zeros(mask.shape)

    if top_cnt is None:
        top_cnt = len(vc)
    top_cnt = min(top_cnt, len(vc))

    for i in range(top_cnt):
        if vc[i][1] < min_vol: break
        if float(vc[i][1]) / total_size < volume_ratio: break
        mask[labels == vc[i][0]] = 1

    return mask


def get_top_components_fast(mask, top_cnt=None, volume_ratio=0, min_vol=1):
    # Find the top largest connected components
    # The algorithm has an implicit assumption: the component has relatively
    # simple structure.
    labels = measure.label(mask, background=0)
    allcounts = np.bincount(labels.flat)
    allcounts[0] = 0
    if top_cnt is None: top_cnt = len(allcounts)
    selected_labels = list(np.argsort(allcounts)[::-1][:top_cnt])
    bounds = scipy.ndimage.find_objects(labels)

    total_size = np.sum(mask)
    new_mask = np.zeros(mask.shape)
    patches = []
    for bound in bounds:
        curr = labels[bound]
        count = np.bincount(curr.flat)
        for label in range(len(count)):
            if count[label] == allcounts[label] and label in selected_labels:
                break
        else:
            # print label
            # print allcounts
            # print count
            # print selected_labels
            # print '-------------------------'
            continue
        selected_labels.remove(label)

        curr_mask = (curr == label).astype(int)
        vol = float(np.sum(curr_mask))
        if vol < min_vol: continue
        if vol / total_size < volume_ratio: continue
        patches.append([curr_mask, bound])
        new_mask[bound] += curr_mask
    new_mask = (new_mask > 0).astype(int)
    return new_mask, patches


def calHessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x)
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian


def calHessianEigen3DFull(image3D, scale):
    image = scipy.ndimage.filters.gaussian_filter(image3D, scale, truncate=10)
    h = calHessian(image)
    h = np.transpose(h, [2, 3, 4, 0, 1])

    e, V = LA.eig(h)

    idx = np.argsort(np.abs(e), axis=3)

    e1 = e[np.arange(len(idx)), idx[:, 0]].reshape([-1, 1])
    e2 = e[np.arange(len(idx)), idx[:, 1]].reshape([-1, 1])
    e3 = e[np.arange(len(idx)), idx[:, 2]].reshape([-1, 1])
    e = np.concatenate([e1, e2, e3], axis=3)

    vec1 = V[np.arange(len(idx)), :, idx[:, 0]].reshape([len(idx), 1, 3])
    vec2 = V[np.arange(len(idx)), :, idx[:, 1]].reshape([len(idx), 1, 3])
    vec3 = V[np.arange(len(idx)), :, idx[:, 2]].reshape([len(idx), 1, 3])
    V = np.concatenate([vec1, vec2, vec3], axis=3)

    return h.astype(float), full_e.astype(float), full_V.astype(float)


def calHessianEigen3D(image3D, scale, mask3D):
    """ Calculate Hessian using whole 3D image but calculate eigen vector
    for selected points for speed """
    image = scipy.ndimage.filters.gaussian_filter(image3D, scale, truncate=10)
    h = calHessian(image)
    h = np.transpose(h, [2, 3, 4, 0, 1])

    selected_points = mask3D.flatten() > 0

    e, V = LA.eig(h.reshape([-1, 3, 3])[selected_points])

    idx = np.argsort(np.abs(e), axis=1)

    e1 = e[np.arange(len(idx)), idx[:, 0]].reshape([-1, 1])
    e2 = e[np.arange(len(idx)), idx[:, 1]].reshape([-1, 1])
    e3 = e[np.arange(len(idx)), idx[:, 2]].reshape([-1, 1])
    e = np.concatenate([e1, e2, e3], axis=1)

    vec1 = V[np.arange(len(idx)), :, idx[:, 0]].reshape([len(idx), 1, 3])
    vec2 = V[np.arange(len(idx)), :, idx[:, 1]].reshape([len(idx), 1, 3])
    vec3 = V[np.arange(len(idx)), :, idx[:, 2]].reshape([len(idx), 1, 3])
    V = np.concatenate([vec1, vec2, vec3], axis=1)

    full_e = np.zeros(selected_points.shape + (3,))
    full_V = np.zeros(selected_points.shape + (3, 3))

    full_e[selected_points] = e
    full_V[selected_points] = V

    full_e = full_e.reshape(image3D.shape + (3,))
    full_V = full_V.reshape(image3D.shape + (3, 3))

    return h.astype(float), full_e.astype(float), full_V.astype(float)


def calVBTScore3D(e, alpha=0.5, beta=0.5, gamma=None, rab_cutoff=0.6):
    # Calculate vesselness, blobness, and tubeness score
    flattend_e = e.reshape([-1, 3])
    idx = np.abs(flattend_e[:, 2]) > 0.001
    flattend_e = flattend_e[idx]

    # abs(lambda1) < abs(lambda2) < abs(lambda3)
    lambda1 = flattend_e[:, 0]
    lambda2 = flattend_e[:, 1]
    lambda3 = flattend_e[:, 2]
    vol = np.sqrt(np.sum(flattend_e ** 2, axis=1))
    ra = np.abs(lambda2) / (np.abs(lambda3) + 1e-10)
    # rb = np.abs(lambda1) / (np.sqrt(np.abs(lambda2 * lambda3)) + 1e-10)
    rb = np.abs(lambda1) / (np.abs(lambda2) + 1e-10)
    part1 = 1 - np.exp(-ra ** 2 / (2 * alpha ** 2))
    part2 = np.exp(-rb ** 2 / (2 * beta ** 2))
    part3 = 1 - np.exp(-vol ** 2 / (2 * gamma ** 2))

    vscore = part1 * part2 * part3 * (lambda2 < 0) * (lambda3 < 0) * (ra > rab_cutoff) * (rb < (1 - rab_cutoff))
    bscore = part1 * (1 - part2) * part3 * (lambda2 < 0) * (lambda3 < 0) * (ra > rab_cutoff) * (ra > rab_cutoff)
    tscore = part1 * part2 * part3 * (lambda2 > 0) * (lambda3 > 0)

    vesselnessScore = np.zeros(len(idx))
    vesselnessScore[idx] = vscore
    vesselnessScore = vesselnessScore.reshape(e.shape[:-1])

    blobnessScore = np.zeros(len(idx))
    blobnessScore[idx] = bscore
    blobnessScore = blobnessScore.reshape(e.shape[:-1])

    tubenessScore = np.zeros(len(idx))
    tubenessScore[idx] = tscore
    tubenessScore = tubenessScore.reshape(e.shape[:-1])

    volume = np.zeros(len(idx))
    volume[idx] = vol
    volume = volume.reshape(e.shape[:-1])

    vesselnessScore = vesselnessScore / (np.max(vesselnessScore) + 1e-10)
    blobnessScore = blobnessScore / (np.max(blobnessScore) + 1e-10)
    tubenessScore = tubenessScore / (np.max(tubenessScore) + 1e-10)

    return vesselnessScore, blobnessScore, tubenessScore, volume


def calMultiScaleScore3D(hessianEigens, alpha=0.1, beta=0.1, gamma=10, rab_cutoff=0.6):
    """ Calculate the max vessel score of different scale"""
    vesselnessScore = 0
    blobnessScore = 0
    tubenessScore = 0
    for h, e, V in hessianEigens:
        currv, currb, currt, volume = calVBTScore3D(e, alpha, beta, gamma, rab_cutoff)
        vesselnessScore = np.maximum(currv, vesselnessScore)
        blobnessScore = np.maximum(currb, blobnessScore)
    return vesselnessScore, blobnessScore, tubenessScore


def calVesselBlob(wholeLungCT3D):
    lungmask, bodymask = seg_lung(wholeLungCT3D)
    lung, lungmask, _, xystarts, _ = extract_lung_region(wholeLungCT3D, lungmask)
    xyslice = [slice(xystarts[0], xystarts[0] + lung.shape[0]), slice(xystarts[1], xystarts[1] + lung.shape[1])]
    bodymask = bodymask[xyslice]
    lungmask = (1 - bodymask) * scipy.ndimage.binary_dilation(lungmask, iterations=10)
    vessels = (lung > -700) * lungmask
    lung = normalize(lung)

    scales = [0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7]
    hessianEigens = []
    for scale in scales:
        hessianEigens.append(calHessianEigen3D(lung, scale=scale, mask3D=vessels))
    vesselnessScore, blobnessScore, tubenessScore = calMultiScaleScore3D(hessianEigens, alpha=0.1, beta=0.1, gamma=10)

    vesselRegion = np.zeros(wholeLungCT3D.shape)
    vesselRegion[xyslice] = vessels
    vesselScore = np.zeros(wholeLungCT3D.shape)
    vesselScore[xyslice] = vesselnessScore - 0.2 * blobnessScore
    blobScore = np.zeros(wholeLungCT3D.shape)
    blobScore[xyslice] = blobnessScore - 0.2 * vesselnessScore
    return vesselRegion, vesselScore, blobScore


def calTube(wholeLungCT3D):
    lungmask, bodymask = seg_lung(wholeLungCT3D)
    lung, lungmask, _, xystarts, _ = extract_lung_region(wholeLungCT3D, lungmask)
    xyslice = [slice(xystarts[0], xystarts[0] + lung.shape[0]), slice(xystarts[1], xystarts[1] + lung.shape[1])]
    bodymask = bodymask[xyslice]
    lungmask = (1 - bodymask) * scipy.ndimage.binary_dilation(lungmask, iterations=10)
    tubes = (lung > -900) * lungmask
    lung = normalize(lung)

    scales = [1, 2, 3, 4, 5, 6, 7]
    hessianEigens = []
    for scale in scales:
        hessianEigens.append(calHessianEigen3D(lung, scale=scale, mask3D=tubes))
    vesselnessScore, blobnessScore, tubenessScore = calMultiScaleScore3D(hessianEigens, alpha=0.1, beta=0.1, gamma=10)

    tubeRegion = np.zeros(wholeLungCT3D.shape)
    tubeRegion[xyslice] = tubes
    tubeScore = np.zeros(wholeLungCT3D.shape)
    tubeScore[xyslice] = tubenessScore
    return tubeRegion, tubeScore


def getAirTube(lung, mask, cutoff=-950):
    """ Get the air tube from CT using a very simple method.
    Note, this method may fail on serious COPD patient """
    tube = ((lung < -950) * mask).astype(float)
    tube = scipy.ndimage.filters.gaussian_filter(tube, 1, truncate=5)
    tube = (tube > 0.5).astype(float)
    labels = measure.label(tube, background=0)
    vals, counts = np.unique(labels, return_counts=True)
    vc = sorted(zip(vals, counts), reverse=True, key=lambda x: x[1] * (x[0] > 0))
    tube = labels == vc[0][0]
    tube = scipy.ndimage.binary_dilation(tube, iterations=3)
    return tube


def calHessianEigen2D(image2D, scale, mask2D):
    """ Calculate Hessian using whole 3D image but calculate eigen vector
    for selected points for speed """
    image = scipy.ndimage.filters.gaussian_filter(image2D, scale, truncate=10)
    h = calHessian(image)
    h = np.transpose(h, [2, 3, 0, 1])

    selected_points = mask2D.flatten() > 0

    e, V = LA.eig(h.reshape([-1, 2, 2])[selected_points])

    idx = np.argsort(np.abs(e), axis=1)

    e1 = e[np.arange(len(idx)), idx[:, 0]].reshape([-1, 1])
    e2 = e[np.arange(len(idx)), idx[:, 1]].reshape([-1, 1])
    e = np.concatenate([e1, e2], axis=1)

    vec1 = V[np.arange(len(idx)), :, idx[:, 0]].reshape([len(idx), 1, 2])
    vec2 = V[np.arange(len(idx)), :, idx[:, 1]].reshape([len(idx), 1, 2])
    V = np.concatenate([vec1, vec2], axis=1)

    full_e = np.zeros(selected_points.shape + (2,))
    full_V = np.zeros(selected_points.shape + (2, 2))

    full_e[selected_points] = e
    full_V[selected_points] = V

    full_e = full_e.reshape(image2D.shape + (2,))
    full_V = full_V.reshape(image2D.shape + (2, 2))

    return h.astype(float), full_e.astype(float), full_V.astype(float)


def calVesselnessScore2D(e, alpha=0.5, beta=0.5):
    """ Example usage:
    h, e, V = calHessianEigen2D(image, 5, mask)
    vess, vol = calVesselnessScore2D(e, alpha = 0.5, beta = 0.5)
    """
    # Calculate vesselness score
    flattend_e = e.reshape([-1, 2])
    idx = np.abs(flattend_e[:, 1]) > 1e-30
    flattend_e = flattend_e[idx]

    lambda1 = flattend_e[:, 0]
    lambda2 = flattend_e[:, 1]

    vol = np.sqrt(np.sum(flattend_e ** 2, axis=1))
    ra = np.abs(lambda2) / (np.abs(lambda1) + 1e-10)
    part1 = 1 - np.exp(-ra ** 2 / (2 * alpha ** 2))
    part2 = 1 - np.exp(-vol ** 2 / (2 * beta ** 2))

    vscore = part1 * part2 * (lambda2 < 0)

    vesselnessScore = np.zeros(len(idx))
    vesselnessScore[idx] = vscore
    vesselnessScore = vesselnessScore.reshape(e.shape[:-1])

    volume = np.zeros(len(idx))
    volume[idx] = np.abs(lambda1)
    volume = volume.reshape(e.shape[:-1])

    vesselnessScore = vesselnessScore / (np.max(vesselnessScore) + 1e-30)
    volume = volume / (np.max(volume) + 1e-30)

    return vesselnessScore, volume


def calculate_density1(image3D, mask3D, width):
    """ Calculate average density """
    kernel = np.ones((width, width, width))
    mask3D = (image3D < -700) * mask3D
    signal = ndimage.convolve(image3D, kernel)
    counter = ndimage.convolve(mask3D, kernel)
    signal = (signal / (1e-10 + counter)) * mask3D
    return signal


def calculate_density_torch(image3D, mask3D, width):
    """ Calculate average density using pytorch and GPU """
    kernel = np.ones((1, 1, width, width, width))
    mask3D = (image3D < -700) * mask3D
    image3D = image3D * mask3D
    image3D = image3D.reshape((1, 1) + image3D.shape)
    mask3D = mask3D.reshape((1, 1) + mask3D.shape)

    kernel = Variable(torch.from_numpy(kernel.astype(np.float32)).cuda())
    image3D = Variable(torch.from_numpy(image3D.astype(np.float32)).cuda())
    mask3D = Variable(torch.from_numpy(mask3D.astype(np.float32)).cuda())
    padding = width / 2
    image3D = F.conv3d(image3D, kernel, padding=padding)
    mask3D = F.conv3d(mask3D, kernel, padding=padding)

    signal = image3D / (1e-10 + mask3D)
    mask3D = (mask3D >= 1).float()

    signal = signal * mask3D
    signal = signal.cpu().data.numpy()
    signal = signal.reshape(signal.shape[2:])
    return signal


def sort_points_by_angle(points, center):
    """ Project the curve to the xy plane and sort by center
    points is in shape 2xN
    """
    center = np.array(center)
    center = center.reshape(2, 1)
    diff = points[:2] - center
    length = np.sqrt(np.sum(diff ** 2, axis=0, keepdims=True))
    angle1 = np.arccos(diff[0] / length).flatten()
    angle2 = np.arcsin(diff[1] / length).flatten()
    angle1 = (np.pi * 2 - angle1) * (angle2 < 0) + angle1 * (angle2 > 0)
    idx = np.argsort(angle1)
    angle1 = angle1[idx]
    points = points[:, idx]

    # If the curve transpase the positive x-axis
    diff = angle1[1:] - angle1[:-1]
    idx = np.argmax(diff)
    # print
    # diff[idx]
    if diff[idx] > 1:
        idx += 1
        angle1 = np.concatenate([angle1[idx:], angle1[:idx] + 2 * np.pi])
        points = np.concatenate([points[:, idx:], points[:, :idx]], axis=1)

    return angle1, points


def rotate_origin_only(xy, radians):
    """Only rotate a point around the origin (0, 0)."""
    x, y = xy
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)
    return np.array([xx, yy])


def extract_patches(image3D, points3D, window, min_dist=1.0):
    """
    For curve defined by a series point extract local patches centered
    around each point and the patch is parallel to the z-axis and
    perpendicular to the direction of the curve

    Make sure that the extracted window is in the boundary of image3D
    points3D is in shape [C, 3].
    """
    patches = []
    coords = []
    for i in range(1, len(points3D)):
        diff = points3D[i].astype(int) - points3D[i - 1].astype(int)
        if np.sum(diff * diff) == 0:
            continue

        # Take only 2D direction
        direction = points3D[i] - points3D[i - 1]
        direction = rotate_origin_only(direction[:2], np.pi * 1.5)
        direction = direction / np.sqrt(np.sum(direction * direction))

        xs = np.arange(-window, window + 1) * direction[0] + points3D[i, 0]
        ys = np.arange(-window, window + 1) * direction[1] + points3D[i, 1]
        zs = np.arange(-window, window + 1) + points3D[i, 2]

        xs = np.repeat(xs.reshape(1, -1), window * 2 + 1, axis=0).flatten()
        ys = np.repeat(ys.reshape(1, -1), window * 2 + 1, axis=0).flatten()
        zs = np.repeat(zs.reshape(-1, 1), window * 2 + 1, axis=1).flatten()

        patch = util.interpolate_cubic(image3D, xs, ys, zs)

        # patch = image3D[np.round(xs).astype(int), np.round(ys).astype(int), np.round(zs).astype(int)]
        patch = patch.reshape(window * 2 + 1, window * 2 + 1)
        xs = xs.reshape(window * 2 + 1, window * 2 + 1)
        ys = ys.reshape(window * 2 + 1, window * 2 + 1)
        zs = zs.reshape(window * 2 + 1, window * 2 + 1)

        patches.append([patch, (xs, ys, zs)])
        coords.append(points3D[i])

    patches = np.array(patches)
    coords = np.array(coords)
    coords[:, 0] = np.clip(coords[:, 0], 0, image3D.shape[0] - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, image3D.shape[1] - 1)
    coords[:, 2] = np.clip(coords[:, 2], 0, image3D.shape[2] - 1)
    return patches, coords


def polar_unfold(body, centers=None, step=0.01, sangle=np.pi * 1.5, eangle=np.pi * 3.5):
    """ Generate a 2D flattend image """
    if centers is None:
        centers = []
        xs = []
        for i in range(body.shape[2] - 5):
            projection = np.max(body[:, :, i:i + 5] > 0, axis=2)
            objs = scipy.ndimage.find_objects(projection)
            if len(objs) == 0: continue
            slicex, slicey = scipy.ndimage.find_objects(projection)[0]
            center = ((slicex.stop + slicex.start) / 2.0, (slicey.stop + slicey.start) / 2.0)
            centers.append(center)
            xs.append(i)
        centers = np.array(centers) - (0, 50)
        xs = np.array(xs)
        x = util.smooth_curve_1d(xs, centers[:, 0], range(body.shape[2]), 2)
        y = util.smooth_curve_1d(xs, centers[:, 1], range(body.shape[2]), 2)
        centers = np.array(zip(x, y)).astype(int)

    polar_body = []
    coords = []
    for i in range(body.shape[2]):
        polar, cc = util.cart2polar2D(body[:, :, i], center=centers[i], step=step, sangle=sangle, eangle=eangle)
        polar_body.append(polar)
        coords.append(cc)
    polar_body = np.array(polar_body)  # [Z, R, Angle]
    coords = np.array(coords)  # [Z, 2, R*Angle]
    polar_body_2D = np.max(polar_body, axis=1)
    return polar_body_2D, polar_body, coords, centers


def polar_deunfold(polar_body, coords, cart_shape):
    """ Given the output of polar_unfold, do the reverse operation """
    body = []
    for i in range(polar_body.shape[0]):
        body.append(util.polar2cart2D(polar_body[i], coords[i], cart_shape)[0])
    body = np.array(body)
    body = np.transpose(body, [1, 2, 0])
    return body


def get_soft_bone(lung):
    """ Get the MIP of soft bone """
    slicex, slicey, slicez = scipy.ndimage.find_objects(lung > 0)[0]
    endy = (slicey.stop + slicey.start) / 2 - 50
    starty = slicey.start

    body_part = lung[:, starty:endy]
    body_part = np.clip(body_part, -450, 600)

    max_idx = np.argmax(body_part, axis=1)
    body_part = np.max(body_part, axis=1)
    return body_part, max_idx


def add_poisson_noise(image, level):
    """ image ranges from [0, 1], level range from [0, inf]. small level means large noise.
    Suggested value for level is [25, 255]"""
    image = np.random.poisson(image * level) / float(level)
    image = np.clip(image, 0, 1)
    return image


def add_poisson_noise_HU(cube):
    cube = np.clip(cube, -1000, 1000) + 1000
    level = random.uniform(-1, 1)
    level = 10 ** level  # level ranges from 10 ~ 0.1
    cube = add_poisson_noise(cube, level)
    cube = np.clip(cube - 1000, -1000, 1000)
    return cube


def add_gaussian_noise_HU(cube):
    std = random.uniform(0.00001, 0.03 * 1450)
    cube = cube + np.random.normal(0, std, size=cube.shape)
    return cube


# def debug():
#     x = [(1, 2, 3), (10, 20, 4), (100, 200, 10)]
#     y = [(1, 5, 10), (10, 21, 12), (101, 202, 16), (20, 20, 12)]
#
#     print
#     x
#     print
#     y
#     print
#     alignment_nodules(x, y)
