import numpy as np
from scipy.ndimage import geometric_transform

a = np.arange(12.).reshape((4, 3))


def shift_func(output_coords):
    return (output_coords[0] - 0.0, output_coords[1] - 1.0)

print(a)
print(geometric_transform(a, shift_func))
