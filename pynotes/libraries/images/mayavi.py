from mayavi import mlab

pts = mlab.points3d(xdata, ydata, zdata, sdata, scale_mode='none', scale_factor=0.8, colormap='jet')
mlab.show()
