import os
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
niftifolder = args.srcdir
nrrdfolder = args.dstdir

nrrd_ext = '.nrrd'


def cvt_nii_to_nrrd(name):
    print(name)
    niftipath = os.path.join(niftifolder, name)
    filebase = os.path.splitext(name)[0]
    filebase = os.path.splitext(filebase)[0]
    nrrdbasename = filebase + nrrd_ext
    nrrdpath = os.path.join(nrrdfolder, nrrdbasename)

    vol_file = nib.load(niftipath)
    data = vol_file.get_data()

    nrrd.write(nrrdpath, data)


src_list = os.listdir(args.srcdir)

if args.parallel:
    Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(
        delayed(cvt_nii_to_nrrd)(filename) for filename in src_list)
else:
    for filename in src_list:
        cvt_nii_to_nrrd(filename)
