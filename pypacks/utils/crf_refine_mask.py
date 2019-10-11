import numpy as np
# import util
import pypacks.utils.HH_util as util
# import ct_image_util
from pypacks.utils import ct_image_util
# reload(util)
# import skimage
import scipy.ndimage
import scipy.ndimage.morphology


# def refine_aorta_mask(lung, mask, deltaHU=30, init_std=3, iterations=30,
#                       neigh=(3, 3), neigh_sigmas=(5, 5), weight=5, comp_std=.1):
#     lung = np.concatenate([np.repeat(lung[0:1], 5, axis=0), lung, np.repeat(lung[-1:], 5, axis=0)])
#     mask = np.concatenate([np.repeat(mask[0:1], 5, axis=0), mask, np.repeat(mask[-1:], 5, axis=0)])
#
#     lung_polar = []
#     mask_polar = []
#     fixed_location = []
#     mask_radius = []
#     bound_prob = []
#     coords = []
#     centers = []
#
#     def diff(image):
#         dummy = np.array([0] * image.shape[1])
#         dummy = dummy.reshape([1, -1])
#         image = np.concatenate([dummy, image[:-1] - image[1:]])
#         return image
#
#     signal = lung.flatten()[mask.astype(bool).flatten()]
#     print(np.std(signal), np.mean(signal))
#     HU = np.median(signal) - 30  # np.std(signal)
#     print(HU)
#     for z in range(len(lung)):
#         c = np.array(scipy.ndimage.measurements.center_of_mass(mask[z])).astype(int)
#         centers.append(c)
#
#         init_boundary = util.cart2polar2D(mask[z].astype(float), center=c, r=30, eangle=8)[0]
#         mask_polar.append(init_boundary)
#         init_boundary = np.maximum(0, diff(init_boundary))
#         init_boundary = scipy.ndimage.filters.gaussian_filter(init_boundary.astype(float), init_std)
#         init_boundary = init_boundary / (float(np.max(init_boundary)) + 1e-10)
#
#         curr, coord = util.cart2polar2D(lung[z], center=c, r=30, eangle=8)
#         curr_mask = np.clip(curr, -1000, 100)
#         curr_mask = np.maximum(0, diff(curr_mask))
#         curr_mask = (curr_mask > 100) * (init_boundary > 0.1)
#         curr_mask = ct_image_util.get_top_components(curr_mask, top_cnt=1)
#         curr_mask = np.repeat(np.max(curr_mask, axis=0, keepdims=True), curr_mask.shape[0], axis=0)
#         fixed_location.append(curr_mask)
#
#         curr = np.clip(curr, HU - deltaHU, HU + 100)
#         lung_polar.append(curr)
#         coords.append(coord)
#         curr = np.maximum(0, diff(curr))
#         grad = 1.0 / np.arange(1, curr.shape[0] + 1)
#         grad = grad.reshape([-1, 1])
#         curr = (curr ** 4) * init_boundary * (grad ** 4)
#         curr = (curr + 1e-10) / (np.sum(curr, axis=0, keepdims=True) + 1e-10)
#         bound_prob.append(curr)
#
#         mask_radius.append(curr)
#
#     # lung_polar = np.array(lung_polar)
#     # mask_polar = np.array(mask_polar)
#     # bound_prob = np.array(bound_prob)
#     mask_radius = np.array(mask_radius)
#     fixed_location = np.array(fixed_location).astype(int)
#
#     mask_refined = mask_radius.transpose([1, 0, 2])
#     fixed_location = fixed_location.transpose([1, 0, 2])
#     mask_refined = util.mask_refiner(mask_refined, mask=fixed_location, iterations=iterations, neigh=neigh,
#                                      neigh_sigmas=neigh_sigmas, weight=weight, comp_std=comp_std)
#     mask_refined = mask_refined.transpose([1, 0, 2])
#     fixed_location = fixed_location.transpose([1, 0, 2])
#
#     #     fig, ax = plt.subplots(6,1, figsize=(50, 10), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
#     #     ax[0].imshow(lung_polar[idx], cmap = plt.get_cmap('gray'))
#     #     ax[1].imshow(mask_polar[idx], cmap = plt.get_cmap('gray'))
#     #     ax[2].imshow(bound_prob[idx], cmap = plt.get_cmap('gray'))
#     #     ax[3].imshow(mask_radius[idx], cmap = plt.get_cmap('gray'))
#     #     ax[4].imshow(fixed_location[idx], cmap = plt.get_cmap('gray'))
#     #     ax[5].imshow(mask_refined[idx], cmap = plt.get_cmap('gray'))
#     #     plt.show()
#
#     # convert polar to card
#     new_mask = []
#     for z in range(len(mask_refined)):
#         curr = (1 - np.cumsum(mask_refined[z], axis=0)) > 0.5
#         curr = scipy.ndimage.binary_dilation(curr, iterations=1)
#         new_mask.append(util.polar2cart2D(curr, coords[z], lung[0].shape)[0])
#     new_mask = np.array(new_mask) > 0.5
#     new_mask = scipy.ndimage.morphology.binary_closing(new_mask)
#
#     new_mask = new_mask[5:-5]
#     return new_mask
#
# #
# # new_mask = refine_aorta_mask(lung, mask, init_std=3, neigh=(3, 3), weight=5)
# # new_mask = refine_aorta_mask(lung, new_mask, init_std=2, neigh=(2, 2), weight=3)
# # new_mask = refine_aorta_mask(lung, new_mask, init_std=1, neigh=(1, 1), weight=1)


def refine_aorta_mask(lung, mask, deltaHU=30, init_std=3, iterations=30,
                      neigh=(3, 3), neigh_sigmas=(5, 5), weight=5, comp_std=.1):
    lung = np.concatenate([np.repeat(lung[0:1], 5, axis=0), lung, np.repeat(lung[-1:], 5, axis=0)])
    mask = np.concatenate([np.repeat(mask[0:1], 5, axis=0), mask, np.repeat(mask[-1:], 5, axis=0)])

    lung_polar = []
    mask_polar = []
    fixed_location = []
    mask_radius = []
    bound_prob = []
    coords = []
    centers = []

    def diff(image):
        dummy = np.array([0] * image.shape[1])
        dummy = dummy.reshape([1, -1])
        image = np.concatenate([dummy, image[:-1] - image[1:]])
        return image

    signal = lung.flatten()[mask.astype(bool).flatten()]
    print(np.std(signal), np.mean(signal))
    HU = np.median(signal) - 30  # np.std(signal)
    print(HU)
    for z in range(len(lung)):
        c = np.array(scipy.ndimage.measurements.center_of_mass(mask[z])).astype(int)
        centers.append(c)

        init_boundary = util.cart2polar2D(mask[z].astype(float), center=c, r=30, eangle=8)[0]
        mask_polar.append(init_boundary)
        init_boundary = np.maximum(0, diff(init_boundary))
        init_boundary = scipy.ndimage.filters.gaussian_filter(init_boundary.astype(float), init_std)
        init_boundary = init_boundary / (float(np.max(init_boundary)) + 1e-10)

        curr, coord = util.cart2polar2D(lung[z], center=c, r=30, eangle=8)
        curr_mask = np.clip(curr, -1000, 100)
        curr_mask = np.maximum(0, diff(curr_mask))
        curr_mask = (curr_mask > 100) * (init_boundary > 0.1)
        curr_mask = ct_image_util.get_top_components(curr_mask, top_cnt=1)
        curr_mask = np.repeat(np.max(curr_mask, axis=0, keepdims=True), curr_mask.shape[0], axis=0)
        fixed_location.append(curr_mask)

        curr = np.clip(curr, HU - deltaHU, HU + 100)
        lung_polar.append(curr)
        coords.append(coord)
        curr = np.maximum(0, diff(curr))
        #         grad = 1.0/np.arange(1, curr.shape[0]+1)
        #         grad = grad.reshape([-1, 1])
        curr = (curr ** 4) * init_boundary  # * (grad ** 4)
        curr = (curr + 1e-10) / (np.sum(curr, axis=0, keepdims=True) + 1e-10)
        bound_prob.append(curr)

        mask_radius.append(curr)

    lung_polar = np.array(lung_polar)
    mask_polar = np.array(mask_polar)
    bound_prob = np.array(bound_prob)
    mask_radius = np.array(mask_radius)
    fixed_location = np.array(fixed_location).astype(int)

    mask_refined = mask_radius.transpose([1, 0, 2])
    fixed_location = fixed_location.transpose([1, 0, 2])
    mask_refined = util.mask_refiner(mask_refined, mask=fixed_location, iterations=iterations, neigh=neigh,
                                     neigh_sigmas=neigh_sigmas, weight=weight, comp_std=comp_std)
    mask_refined = mask_refined.transpose([1, 0, 2])
    fixed_location = fixed_location.transpose([1, 0, 2])

    #     fig, ax = plt.subplots(6,1, figsize=(50, 10), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
    #     ax[0].imshow(lung_polar[idx], cmap = plt.get_cmap('gray'))
    #     ax[1].imshow(mask_polar[idx], cmap = plt.get_cmap('gray'))
    #     ax[2].imshow(bound_prob[idx], cmap = plt.get_cmap('gray'))
    #     ax[3].imshow(mask_radius[idx], cmap = plt.get_cmap('gray'))
    #     ax[4].imshow(fixed_location[idx], cmap = plt.get_cmap('gray'))
    #     ax[5].imshow(mask_refined[idx], cmap = plt.get_cmap('gray'))
    #     plt.show()

    # convert polar to card
    new_mask = []
    for z in range(len(mask_refined)):
        curr = (1 - np.cumsum(mask_refined[z], axis=0)) > 0.5
        curr = scipy.ndimage.binary_dilation(curr, iterations=1)
        new_mask.append(util.polar2cart2D(curr, coords[z], lung[0].shape)[0])
    new_mask = np.array(new_mask) > 0.5
    new_mask = scipy.ndimage.morphology.binary_closing(new_mask)

    new_mask = new_mask[5:-5]
    return new_mask


# # idx = 20
# new_mask = refine_aorta_mask(lung, mask, init_std=3, neigh=(3, 3), weight=5, comp_std=5)
# new_mask = refine_aorta_mask(lung, new_mask, init_std=2, neigh=(2, 2), weight=3, comp_std=3)
# new_mask = refine_aorta_mask(lung, new_mask, init_std=1, neigh=(1, 1), weight=1, comp_std=1)
