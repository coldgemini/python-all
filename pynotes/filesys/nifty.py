import nibabel as nib

vol_file = nib.load(src)
data = vol_file.get_data()

# ref_affine = np.eye(4)
new_vol_file = nib.Nifti1Image(nparr, vol_file.affine)
nib.save(new_vol_file, nifti_output_path)
