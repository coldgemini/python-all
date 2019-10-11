# import matplotlib

# matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import numpy as np
from functools import partial


def sigmoid0(x, mu=0.0, s=1.0, l=0.0, h=1.0):
    y = l + (h - l) / (1 + np.exp(-s * (x - mu)))
    return y


mu = 0.0
s = 100.0
l = -100.0
h = 150.0


# sigmoid1 = partial(sigmoid0, mu=mu, l=l, h=h)


def equalize(x, mu=mu, s=s, l=l, h=h):
    y = np.piecewise(x, [x < l, np.logical_and(l <= x, x <= h), x > h],
                     [lambda x: x, partial(sigmoid0, s=s, mu=mu, l=l, h=h), lambda x: x])
    return y


s1 = 0.02
s2 = 0.05
s3 = 0.1
a = np.linspace(-500, 500, 200)
b1 = equalize(a, s=s1)
b2 = equalize(a, s=s2)
b3 = equalize(a, s=s3)

# plt.plot(a, b1, color='red', marker="o")
plt.plot(a, b1, label='s={}'.format(s1))
plt.plot(a, b2, label='s={}'.format(s2))
plt.plot(a, b3, label='s={}'.format(s3))
plt.title("equalize")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

plt.show()
