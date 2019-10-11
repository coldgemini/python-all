import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2.5, 2.5, 6)
print(type(x))
print(x.shape)
print(np.piecewise(x, [x < 0, x >= 0], [-1, 1]))
print(np.piecewise(x, [x < 0, x >= 0], [lambda x: -x, lambda x: x]))

mu = 0
s = 100
l = -100
h = 100
x = np.linspace(-500, 500, 200)
# y = np.piecewise(x, [x < l, l <= x <= h, x > h], [lambda x: x, lambda x: x, lambda x: x])
y = np.piecewise(x, [x < -100, np.logical_and(-100 < x, x < 100), x > 100], [lambda x: x, lambda x: x, lambda x: x])
# y = np.piecewise(x, [x <= -100, x > 100], [lambda x: x, lambda x: x])
print(y)


def get_score(hu):
    return np.piecewise(hu, [hu < 130, hu >= 130, hu >= 199, hu >= 299, hu >= 400], [0, 1, 2, 3, 4])

def get_score_th(hu, th):
    # return np.piecewise(hu, [hu < th, hu >= th, hu >= 199, hu >= 299, hu >= 400], [0, 1, 2, 3, 4])
    return np.piecewise(hu, [hu < th, hu >= th, hu >= th + 199, hu >= th + 299, hu >= th + 400],
                        [0, 1, 2, 3, 4])

a = np.linspace(0, 1000, 200)
# b1 = get_score(a)
b1 = get_score_th(a, 200)
plt.plot(a, b1, label='calc')
plt.title("calcification")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

plt.show()
