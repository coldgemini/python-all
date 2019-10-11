import numpy as np
from skimage.morphology import ball, dilation
import nrrd
import torch
import torch.nn.functional as F

filters = torch.randn(33, 16, 3, 3, 3)
inputs = torch.randn(20, 16, 50, 10, 20)
output = F.conv3d(inputs, filters)
print(filters.shape)
print(inputs.shape)
print(output.shape)
np_out = output.data.cpu().numpy()
print(type(output))
print(type(np_out))
# print(np_out)

selem = ball(1).astype(np.uint8)
input = np.zeros((5, 5, 5))
input[2, 2, 2] = 1
path = '/home/xiang/Tmp/input.nrrd'
nrrd.write(path, input)
# mask_hl = dilation(selem, selem).astype(np.uint8)
selem = torch.from_numpy(selem).float()
input = torch.from_numpy(input).float()
print(selem.shape)
print(input.shape)
selem = selem.view(1, 1, 3, 3, 3)
input = input.view(1, 1, 5, 5, 5)
print(selem.shape)
print(input.shape)

output = F.conv3d(input, selem, padding=(1, 1, 1))
np_out = output.data.cpu().numpy()
print(np_out.shape)
np_out = np_out.reshape(5, 5, 5)
np_out = np_out.astype(np.uint8)

path = '/home/xiang/Tmp/test.nrrd'
nrrd.write(path, np_out)
