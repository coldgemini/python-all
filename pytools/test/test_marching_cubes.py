import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nrrd
from skimage import measure
from skimage.draw import ellipsoid
from mayavi import mlab

# import sys

# Generate a level set about zero of two identical ellipsoids in 3D
# ellip_base = ellipsoid(6, 10, 16, levelset=True)
# ellip_double = np.concatenate((ellip_base[:-1, ...],
#                                ellip_base[2:, ...]), axis=0)
# print(ellip_double.dtype)
# print(ellip_double.shape)
# nrrd.write('mcubes.nrrd', ellip_double)

# Use marching cubes to obtain the surface mesh of these ellipsoids
# verts, faces, normals, values = measure.marching_cubes_lewiner(ellip_double, 0)

data_path = "/home/xiang/mnt/Data/aorta_seg_data/coronary/lungmask_raw/1.2.156.112605.14038013507713.181219050016.3.PAohbDhK1TP7IgRc69IM1h4riK7TNA.nrrd"
# data_path = "/data2/home/zhouxiangyong/Data/aorta_seg_data/coronary/lungmask_raw/1.2.156.112605.14038013507713.181219050016.3.PAohbDhK1TP7IgRc69IM1h4riK7TNA.nrrd"
mask, _ = nrrd.read(data_path)
verts, faces, normals, values = measure.marching_cubes_lewiner(mask, 0)
print(verts.shape)
print(faces.shape)
print(normals.shape)
print(values.shape)

print(verts[faces].shape)
X = verts[:, 0]
Y = verts[:, 1]
Z = verts[:, 2]
mlab.triangular_mesh(X, Y, Z, faces, color=(1, 0.5, 1), opacity=1.0)
mlab.show()
# sys.exit(0)
# Display resulting triangular mesh using Matplotlib. This can also be done
# with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
#
# # Fancy indexing: `verts[faces]` to generate a collection of triangles
# mesh = Poly3DCollection(verts[faces])
# mesh.set_edgecolor('k')
# ax.add_collection3d(mesh)
#
# ax.set_xlabel("x-axis: a = 6 per ellipsoid")
# ax.set_ylabel("y-axis: b = 10")
# ax.set_zlabel("z-axis: c = 16")
#
# ax.set_xlim(0, 24)  # a = 6 (times two for 2nd ellipsoid)
# ax.set_ylim(0, 20)  # b = 10
# ax.set_zlim(0, 32)  # c = 16
#
# plt.tight_layout()
# plt.show()
