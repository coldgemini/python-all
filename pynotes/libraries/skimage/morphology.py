from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, \

selem = disk(3)
closed = closing(srcmask_data, selem)
