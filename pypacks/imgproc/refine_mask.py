import scipy.ndimage
# import skimage
# import skimage.morphology
import numpy as np


def interpolate_bilinear(image2D, xs, ys):
    d = 1 - 1e-30
    xs0 = np.floor(xs).astype(int)
    xs1 = np.floor(xs + d).astype(int)
    ys0 = np.floor(ys).astype(int)
    ys1 = np.floor(ys + d).astype(int)
    xs0 = np.clip(xs0, 0, image2D.shape[0] - 1)
    xs1 = np.clip(xs1, 0, image2D.shape[0] - 1)
    ys0 = np.clip(ys0, 0, image2D.shape[1] - 1)
    ys1 = np.clip(ys1, 0, image2D.shape[1] - 1)
    values00 = image2D[xs0, ys0]
    values01 = image2D[xs0, ys1]
    values10 = image2D[xs1, ys0]
    values11 = image2D[xs1, ys1]

    values0 = values00 + (values01 - values00) * (ys - ys0)
    values1 = values10 + (values11 - values10) * (ys - ys0)
    values = values0 + (values1 - values0) * (xs - xs0)

    return values


def cart2polar2D(image2D, center=None, r=None, step=0.1, sangle=0, eangle=2 * np.pi):
    if center is None:
        center = np.array(image2D.shape) / 2.0
    center = np.array(center)
    center = center.reshape([2, 1])
    if r is None:
        r = min(image2D.shape[0] / 2.0, image2D.shape[1] / 2.0)
    polar_coord = np.mgrid[0:r, sangle:eangle:step]
    cc = np.zeros(polar_coord.shape)
    cc[0] = polar_coord[0] * np.cos(polar_coord[1])
    cc[1] = polar_coord[0] * np.sin(polar_coord[1])
    old_shape = cc.shape[1:]
    cc = cc.reshape([2, -1]) + center
    cc[0] = np.maximum(0, np.minimum(image2D.shape[0] - 1, cc[0]))
    cc[1] = np.maximum(0, np.minimum(image2D.shape[1] - 1, cc[1]))
    image2D = interpolate_bilinear(image2D, cc[0], cc[1])
    image2D = image2D.reshape(old_shape)
    return image2D, cc


def mask_refiner(prob, iterations=1, neigh=(5, 2, 2), neigh_sigmas=(5, 0.5, 0.5), weight=50, comp_std=0):
    """ Given a 1D/2D/3D mask probability refine it using
    mean field approximation of CRF
    The shape of prob should be channel,layers,X,Y or channel,X,Y or channel,X

    neigh_sigmas = (5,0.5,0.5) means to capture strong dependency between layers
    but weak dependency in the same layer

    weight defines how strong is the neighbors influencing the current location

    If comp_std is 0, then Potts model for the compatability will be used.
    Otherwise, a distance model using gaussian distance will be used,
    comp_std will be used as std. Smaller value means that label i and i+1
    has higher energy
    """
    neigh = np.array(neigh)
    dim = len(neigh)
    neigh_kernel = np.zeros(neigh * 2 + 1)
    neigh_kernel = neigh_kernel.reshape([1] + list(neigh_kernel.shape))
    if dim == 1:
        neigh_kernel[0, neigh[0]] = 1
    elif dim == 2:
        neigh_kernel[0, neigh[0], neigh[1]] = 1
    else:
        neigh_kernel[0, neigh[0], neigh[1], neigh[2]] = 1
    neigh_kernel = scipy.ndimage.filters.gaussian_filter(neigh_kernel, (1,) + neigh_sigmas)
    if dim == 1:
        neigh_kernel[0, neigh[0]] = 0
    elif dim == 2:
        neigh_kernel[0, neigh[0], neigh[1]] = 0
    else:
        neigh_kernel[0, neigh[0], neigh[1], neigh[2]] = 0
    neigh_kernel = neigh_kernel / np.sum(neigh_kernel)

    prob = prob / np.sum(prob, axis=0, keepdims=True)
    log_prob = np.log(prob + 1e-10)
    new_prob = np.zeros(prob.shape)

    for i in range(iterations):
        prob = np.log(prob + 1e-10)
        prob = scipy.ndimage.convolve(prob, neigh_kernel)

        for j in range(prob.shape[0]):
            if comp_std == 0:
                new_prob[j] = log_prob[j] - (np.sum(prob, axis=0) - prob[j]) * weight
            else:
                # The following compatability function has the following effect.
                # It is a generalization of the Potts model
                # If the values of two labels are close, then the related energy is small.
                dist = np.zeros(prob.shape[0])
                dist[j] = 1
                dist = scipy.ndimage.filters.gaussian_filter(dist, comp_std)
                dist = dist / np.max(dist)
                dist = 1 - dist
                # Normalize to avoid that the mass collapse to the first or last label
                # because the boundary label has different total dist sum
                dist = dist / np.sum(dist) * prob.shape[0]

                new_shape = [1] * (dim + 1)
                new_shape[0] = len(dist)
                dist = dist.reshape(new_shape)
                new_prob[j] = log_prob[j] - (np.sum(prob * dist, axis=0)) * weight

        prob = new_prob
        prob = prob - np.max(prob, axis=0, keepdims=True)
        prob = np.exp(prob)
        prob = prob / np.sum(prob, axis=0, keepdims=True)

    return prob
