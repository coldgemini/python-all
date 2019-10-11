npdst = npsrc[:, :, index:index + 3]
data = data[:, :, 0::5]

numpy.concatenate( LIST, axis=0 )
numpy.stack( LIST, axis=0 )
numpy.vstack( LIST )
numpy.array( LIST )
np.stack((a, b), axis=-1)

npz = np.load(numpypath)
print(npz.files)
nparr = npz['arr_0']
print(nparr.shape)

np.savez(slice_path, npz_slice)

list = []
list.append(image)
list.append(image)
batch = np.stack(list)

np.count_nonzero(a == 1)

srcmsk_1d = np.sum(srcmsk_data, axis=(0, 1))

np.where(a < 5, a, 10 * a)

>> > import numpy as np
>> > squarer = lambda t: t ** 2
>> > x = np.array([1, 2, 3, 4, 5])
>> > squarer(x)
array([1, 4, 9, 16, 25])

y = np.expand_dims(x, axis=0)
np.squeeze(x, axis=0).shape

def bbox1(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox


def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def bbox2_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax


import itertools


def bbox2_ND(img):
    N = img.ndim
    out = []
    for ax in itertools.combinations(reversed(range(N)), N - 1):
        nonzero = np.any(img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    return tuple(out)
