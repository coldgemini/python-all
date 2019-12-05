import numpy as np
import h5py

h5file_3d_path = '/data2/home/zhouxiangyong/Workspace/Dev/pytorch-3dunet/resources/random_label3D.h5'
h5file_4d_path = '/home/xiang/Workspace/Dev/pytorch-3dunet/resources/random_label4D.h5'

my_h5file_path = '/data2/home/zhouxiangyong/Data/Heart/unet3d_trial/train/hdf5/ct_train_1001.h5'

h5file_path = '/data2/home/zhouxiangyong/Data/Heart/unet3d_trial/train/hdf5_128/ct_train_1001.h5'
# h5file_path = '/data2/home/zhouxiangyong/Workspace/Data/coronary_data/data/mmwhs_train_test/h5_2p5x/ct_train_1001.h5'
# hf = h5py.File(h5file_3d_path, 'r')
# print(hf.keys())
# n1 = hf.get('raw')
# n1 = np.array(n1)
# print(n1.shape)
# hf.close()
#
# hf = h5py.File(h5file_3d_path, 'r')
# print(hf.keys())
# n1 = hf['raw'][...]
# print(type(n1))
# n1 = np.array(n1)
# print(n1.shape)
# hf.close()

hf = h5py.File(h5file_path, 'r')
# hf = h5py.File(h5file_4d_path, 'r')
# hf = h5py.File(my_h5file_path, 'r')
print(hf.keys())

n1 = hf['raw'][...]
print(type(n1))
n1 = np.array(n1)
print(n1.shape)
print(n1.dtype)
print(np.unique(n1))

n1 = hf['label'][...]
print(type(n1))
n1 = np.array(n1)
print(n1.shape)
print(n1.dtype)
print(np.unique(n1))
hf.close()
