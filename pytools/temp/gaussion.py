import numpy as np
import matplotlib.pyplot as plt

# imglist = os.listdir(imgpath)
# img0 = cv2.imread(imgpath+'/'+imglist[0])
gaussmap = np.zeros(shape=([3, 3]))

h = 3
w = 3
Y, X = np.mgrid[0:h:1, 0:w:1]
# positions = np.vstack([X.ravel(), Y.ravel()])
Y = Y.flatten()
X = X.flatten()
print(Y)
print(X)

gaussmap[Y, X] = np.exp(-((Y - 1) ** 2 + (X - 1) ** 2))

gaussmap = gaussmap.reshape((3, 3))

print(gaussmap)
# for p in xrange(pnum):
#     x = points[p, 0]
#     y = points[p, 1]
#     v = [[0.2 * m, 0], [0, 0.2 * m]]  # standard deviation#协方差为零
#     dx = v[0][0]
#     DX = np.square(dx)
#     dy = v[1][1]
#     DY = np.square(dy)
#     part1 = 1 / (2 * np.pi * dx * dy)
#     p1 = -1.0 / 2
#     px = (X - x) ** 2 / DX
#     py = (Y - y) ** 2 / DY
#     Z = part1 * np.exp(p1 * (px + py))
#     gaussmap += Z
# plt.imshow(gaussmap)
# plt.colorbar()
# plt.show()
# return gaussmap
