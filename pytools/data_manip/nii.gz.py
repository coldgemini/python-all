import nibabel as nib

INPUT_MASK = '/data2/home/zhouxiangyong/Workspace/Dev/AortaSlice/data/raw_aorta_mask_combined_fxc/1.3.6.1.4.1.14519.5.2.1.6279.6001.670107649586205629860363487713.nii.gz'
OUTPUT_MASK = '/data2/home/zhouxiangyong/Tmp/output.nii.gz'

vol_file = nib.load(INPUT_MASK)
data = vol_file.get_data()

img = nib.Nifti1Image(data, vol_file.affine)
nib.save(img, OUTPUT_MASK)
