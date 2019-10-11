import numpy as np
from skimage.measure import label
import nibabel as nib

INPUT_MASK = '/data2/home/zhouxiangyong/Workspace/Dev/AortaSlice/data/raw_aorta_mask_combined_fxc/1.3.6.1.4.1.14519.5.2.1.6279.6001.670107649586205629860363487713.nii.gz'
OUTPUT_MASK = '/data2/home/zhouxiangyong/Tmp/output.nii.gz'


def getLargestCC(segmentation):
    labels = label(segmentation)
    unique, counts = np.unique(labels, return_counts=True)
    list_seg = list(zip(unique, counts))[1:]  # the 0 label is by default background so take the rest
    largest = max(list_seg, key=lambda x: x[1])[0]
    labels_max = (labels == largest).astype(np.uint8)
    return labels_max


vol_file = nib.load(INPUT_MASK)
mask = vol_file.get_data()

mask = getLargestCC(mask).astype(np.uint8)

img = nib.Nifti1Image(mask, vol_file.affine)
nib.save(img, OUTPUT_MASK)
