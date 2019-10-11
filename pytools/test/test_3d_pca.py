import numpy as np
import nrrd
from sklearn.decomposition import PCA

data_path = '/data2/home/zhouxiangyong/Data/aorta_seg_data/coronary/registration/ct_data.nrrd'
mask_path = '/data2/home/zhouxiangyong/Data/aorta_seg_data/coronary/registration/ct_label_embedded.nrrd'
crop_path = '/data2/home/zhouxiangyong/Proj/coronary/data/pca/heart_crop.nrrd'


def bbox_3D(img):
    # ordered by nrrd format (x,y,z)
    # TODO: refactor the x y z order to y x z order
    y = np.any(img, axis=(1, 2))
    x = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    ymin, ymax = np.where(y)[0][[0, -1]]
    xmin, xmax = np.where(x)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return ymin, ymax, xmin, xmax, zmin, zmax


def crop_heart():
    data, _ = nrrd.read(data_path)
    mask, _ = nrrd.read(mask_path)
    ymin, ymax, xmin, xmax, zmin, zmax = bbox_3D(mask)
    print("truncate label results")
    print(xmin, xmax, ymin, ymax, zmin, zmax)
    crop = data[ymin:ymax, xmin:xmax, zmin:zmax]
    nrrd.write(crop_path, crop)


def calc_3d_pca(data):
    crop, _ = nrrd.read(crop_path)
    print(crop.shape)
    pca = PCA(n_components=2)
    slice = crop[:, :, 0]
    pca.fit(slice)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    pass


if __name__ == '__main__':
    # crop_heart()
    # nparr = np.random.rand(3, 3, 3)
    calc_3d_pca()
