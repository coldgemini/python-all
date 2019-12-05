import numpy as np
import h5py

d1 = np.random.random(size=(100, 33))
d2 = np.random.random(size=(100, 333))
d3 = np.random.random(size=(100, 3333))

hf = h5py.File('data.h5', 'w')

g1 = hf.create_group('group1')
g1.create_dataset('data1', data=d1)
g1.create_dataset('data2', data=d1)
g2 = hf.create_group('group2/subfolder')
g2.create_dataset('data3', data=d3)
group2 = hf.get('group2/subfolder')
print(group2.items())
group1 = hf.get('group1')
print(group1.items())
n1 = group1.get('data1')
print(np.array(n1).shape)
n2 = group1.get('data2')
print(np.array(n2).shape)
hf.close()
