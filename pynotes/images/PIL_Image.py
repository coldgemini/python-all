from PIL import Image
import numpy as np

i = Image.open("C:/Users/User/Desktop/mesh.bmp")
i = i.convert("L")
a = np.asarray(i)
b = np.abs(np.fft.rfft2(a))
j = Image.fromarray(b)
j.save("C:/Users/User/Desktop/mesh_trans",".bmp")