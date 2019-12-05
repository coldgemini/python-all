import nrrd
from util.cuda_ops_wrapper import rotate2dL, shift3d

image_path = "/data2/home/zhouxiangyong/Tmp/Debug/cuda/lungcrop.nrrd"
shifted_path = "/data2/home/zhouxiangyong/Tmp/Debug/cuda/shift.nrrd"
rotated_path = "/data2/home/zhouxiangyong/Tmp/Debug/cuda/rotate.nrrd"

image, _ = nrrd.read(image_path)
image_shift = shift3d(image, shift=(50, 100, 0))
image_rotate = rotate2dL(image, angle=45, axis=2)

nrrd.write(shifted_path, image_shift)
nrrd.write(rotated_path, image_rotate)
