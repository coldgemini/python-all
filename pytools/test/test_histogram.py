import numpy as np
import matplotlib.pyplot as plt

a = np.arange(0, 10)
print(a)


plt.hist(a, bins='auto')
plt.title("Histogram with 'auto' bins")
plt.show()
