import numpy as np
import h5py

hf = h5py.File('data.h5', 'r')
print(hf.keys())
n1 = hf.get('dataset_1')
n1 = np.array(n1)
print(n1.shape)
hf.close()
