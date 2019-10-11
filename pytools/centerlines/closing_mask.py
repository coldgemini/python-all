import os
import nrrd
from skimage.morphology import ball, closing
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--srcmask", type=str, help="src mask")
parser.add_argument("-r", "--radius", type=int, default=10, help="disk radius")
parser.add_argument("-n", "--n_jobs", type=int, default=40, help="parallel jobs")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
args = parser.parse_args()
srcmask_dir = args.srcmask
rad = args.radius
n_jobs = args.n_jobs
parallel = args.parallel

filelist = os.listdir(srcmask_dir)


def restore_mask(filename):
    print(filename)

    srcmask_path = os.path.join(srcmask_dir, filename)

    srcmask_data, _ = nrrd.read(srcmask_path)

    selem = ball(rad)
    closed = closing(srcmask_data, selem)

    nrrd.write(srcmask_path, closed)

    return


if parallel:
    Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(restore_mask)(filename) for filename in filelist)
else:
    for filename in filelist:
        restore_mask(filename)
