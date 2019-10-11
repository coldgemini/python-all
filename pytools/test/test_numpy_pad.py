import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.pad(x, ((1, 3), (2, 4)), mode='constant')[1:-3, 2:-4]

print(np.all(x == y))
print(x)
print(y)

print(np.pad(x, ((0, 0), (1, 0)), mode='constant')[:, 0:-1])
