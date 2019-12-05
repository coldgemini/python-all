import os
import numpy as np
import nrrd
import nibabel as nib
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--srcdir", type=str, help="src image folder")
parser.add_argument("-d", "--dstdir", type=str, help="dst image folder")
parser.add_argument("-n", "--n_jobs", type=int, default=20, help="parallel jobs")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
args = parser.parse_args()

nrrdfolder = args.srcdir
niftifolder = args.dstdir


def cvt_nrrd_to_nifti(filename_nrrd):
    print(filename_nrrd, flush=True)
    filename_nifti = filename_nrrd.replace('.nrrd', '.nii.gz')
    nrrd_path = os.path.join(nrrdfolder, filename_nrrd)
    nifti_path = os.path.join(niftifolder, filename_nifti)

    data, _ = nrrd.read(nrrd_path)

    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, nifti_path)


src_list = os.listdir(args.srcdir)

if args.parallel:
    Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(
        delayed(cvt_nrrd_to_nifti)(filename) for filename in src_list)
else:
    for filename in src_list:
        cvt_nrrd_to_nifti(filename)
