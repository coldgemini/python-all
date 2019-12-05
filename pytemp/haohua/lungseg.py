# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import cv2
from skimage import transform
import scipy
from skimage import measure
from scipy.spatial import ConvexHull
import math
import skimage

print(tf.__version__)


class LungSeg(object):
    def __init__(self, sess, model_path, use_bio=True, input_name='input_img:0', output_name='Add_88:0',
                 input_size=(128, 128)):
        self.model_path = model_path
        self.input_size = input_size
        self.sess = sess
        self.input_w = self.input_size[0]
        self.input_h = self.input_size[1]
        self._load_model(self.model_path, use_bio)
        self.input_tensor = self.sess.graph.get_tensor_by_name(input_name)
        self.output_tensor = self.sess.graph.get_tensor_by_name(output_name)

    def _seg_breast_mask(self, image):
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
            try:
                hull = ConvexHull(points)
            except:
                print('Warning Mask')
                return bodymask

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

        return bodymask

    def _edge_smooth(self, mask, lung, lung_threshold=-500):
        mask *= 255
        mask = scipy.ndimage.filters.gaussian_filter(mask, 3, truncate=7)
        mask = (mask > 120).astype(int)
        mask = mask * (lung < lung_threshold)

        return mask.astype(np.uint8)

    def _edge_expansion(self, mask, lung, lung_threshold=-500):
        mask_inter = mask * (lung < lung_threshold)
        lung_binary = (lung < lung_threshold).astype(np.uint8)
        mask_candidate = lung_binary - mask
        new_mask = mask_inter.copy()
        mask_candidate = mask_candidate.transpose((2, 0, 1))
        mask_inter = mask_inter.transpose((2, 0, 1))

        for index, (mi, mc) in enumerate(zip(mask_inter, mask_candidate)):
            mi_contours = measure.find_contours(mi, 0.5)

            mi_contour_mask = np.zeros_like(mi, dtype='bool')
            for point in mi_contours:
                mi_contour_mask[np.round(point[:, 0]).astype('int'), np.round(point[:, 1]).astype('int')] = 1

            conn_mask = measure.label(mc, connectivity=2)
            conn_inter_mask = conn_mask * mi_contour_mask

            point = np.nonzero(conn_inter_mask)
            conn_index = set([conn_inter_mask[x, y] for x, y in zip(point[0], point[1])])

            new_mask_slice = np.zeros_like(mi, dtype='bool')
            for c_i in conn_index:
                new_mask_slice[conn_mask == c_i] = 1

            new_mask[:, :, index] = (mi + new_mask_slice).astype(np.bool).astype(np.int8)

        return new_mask

    def _post_processing(self, masks_xyz):
        z_sum = np.sum(masks_xyz, axis=(0, 1))
        z_max_index = np.where(z_sum == np.max(z_sum))[0][0]
        z_slice = masks_xyz[:, :, z_max_index]

        y_sum = np.sum(z_slice, axis=0)
        y_max_index = np.where(y_sum == np.max(y_sum))[0][0]

        ori_y_dim = z_slice[:, y_max_index].copy()
        mask_y_dim = np.ones(masks_xyz.shape[1])
        masks = masks_xyz.copy()
        masks[:, y_max_index, z_max_index] = mask_y_dim
        masks = measure.label(masks, connectivity=2)
        max_num = 0
        for j in range(1, np.max(masks) + 1):
            if np.sum(masks == j) > max_num:
                max_num = np.sum(masks == j)
                max_pixel = j
            if np.sum(masks == j) > 0.1 * np.sum(masks != 0):
                masks[masks == j] = max_pixel

        masks[masks != max_pixel] = 0
        masks[masks == max_pixel] = 1

        masks[:, y_max_index, z_max_index] &= ori_y_dim

        return masks

    def _load_model(self, model_path, use_bio=True):
        if use_bio:
            from util import zip2bio
            bio = zip2bio.open(model_path)
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(bio.getvalue())
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            if bio is not None:
                bio.close()
        else:
            with tf.gfile.FastGFile(model_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                self.sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')

    def _pre_processing(self, lung_xyz):
        lung_xyz = (np.clip(lung_xyz, -1000, 1000) * 0.1275 + 127.5).astype(np.uint8)
        resized_lung_xyz = transform.resize(lung_xyz.astype(np.float32), self.input_size).astype(np.uint8)
        resized_lung_zxy = resized_lung_xyz.transpose((2, 0, 1))
        return resized_lung_zxy

    def inference(self, lung_xyz, use_post_process=False):
        ori_w = lung_xyz.shape[0]
        ori_h = lung_xyz.shape[1]

        input_lung_zxy = self._pre_processing(lung_xyz)
        mask_zxy = np.zeros(input_lung_zxy.shape)

        for index, input_lung in enumerate(input_lung_zxy):
            input = ((input_lung - 127.5) / 127.5).astype(np.float32)
            output = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: np.array([[input]])})
            output_mask = (output[:, 1, :, :] > output[:, 0, :, :]).astype(np.int8)
            mask_zxy[index, :, :] = output_mask

        mask_xyz = mask_zxy.transpose((1, 2, 0))
        mask_xyz = self._post_processing(mask_xyz.astype(np.uint8))
        mask_xyz = transform.resize(mask_xyz.astype(np.float32), (ori_w, ori_h)).astype(np.uint8)
        mask_xyz = self._edge_smooth(mask_xyz, lung_xyz)
        mask_xyz = self._edge_expansion(mask_xyz, lung_xyz)

        return mask_xyz


if __name__ == '__main__':
    """
    Example: 
    model_path = 'xxx.pb'
    #LungSeg class init params is: tensorflow session and .pb model path
    ls =  LungSeg(tensorflow_session, model_path, use_bio=False)
    #input is a nparray of 3D lung CT image, output is same Dimension nparray of lung seg mask 
    output_mask = ls.inference(lung_3d_xyz) 
    """

    # testing code
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    dir = '../lung_seg_model/lung_seg_0.1.0_tf1.13.pb'
    # dir = './convert/lung_seg_gcn_85_zip.zip'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config=config)
    # init = tf.global_variables_initializer()
    # self.sess.run(init)
    ls = LungSeg(sess, dir, use_bio=False)

    from Utils import Tools
    from Utils import TxtFileTools
    from Utils import ImageTools

    '''
    check_path = './checking/error_manual_image'
    error_list_image = TxtFileTools.read_txt(check_path)
    lung_3d_xyz = np.load(error_list_image[0])['arr_0']
    '''
    lung_3d_xyz = np.load('/data2/fxcdata/image.npy')
    output_mask = ls.inference(lung_3d_xyz)

    np.save('/data2/gmx_data/image.npy', output_mask)
    output_mask = output_mask.transpose((2, 0, 1))
    for index, om in enumerate(output_mask):
        ImageTools.save_image(om * 255, Tools.joint_path('./convert_tf_results', "filename_{}_{}.png".format(0, index)))
