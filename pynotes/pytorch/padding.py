import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image

im = Image.open("/home/xiang/Pictures/heibai.jpg", 'r')

X = torch.Tensor(np.asarray(im))
print("shape:", X.shape)
# dim = (10, 10, 10, 10)
dim = (0, 10, 0, 10)
# X = F.pad(X, dim, "constant", value=0)
X = F.pad(X, dim)

padX = X.data.numpy()
padim = Image.fromarray(padX)
padim = padim.convert("RGB")  # 这里必须转为RGB不然会

padim.save("padded.jpg", "jpeg")
padim.show()
print("shape:", padX.shape)
