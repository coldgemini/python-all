from scipy.ndimage import convolve
import numpy as np

a = np.full((3, 3, 3), 1)
k = np.full((3, 3, 3), 1)
c = convolve(a, k, mode='constant', cval=0.0)
print(c)
