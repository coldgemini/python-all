import numpy as np
import h5py

d1 = np.random.random(size=(100, 33))
d2 = np.random.random(size=(100, 333))

hf = h5py.File('data.h5', 'w')

hf.create_dataset('dataset_1', data=d1, compression="gzip", compression_opts=9)
hf.create_dataset('dataset_2', data=d2, compression="gzip", compression_opts=9)

hf.close()
