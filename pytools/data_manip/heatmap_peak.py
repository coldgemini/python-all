import os
import numpy as np
import scipy.ndimage
from scipy.ndimage.filters import maximum_filter
import nrrd
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--srcdir", type=str, help="src image folder")
parser.add_argument("-d", "--dstdir", type=str, help="dst image folder")
parser.add_argument("-n", "--n_jobs", type=int, default=20, help="parallel jobs")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
args = parser.parse_args()

src_dir = args.srcdir
dst_dir = args.dstdir

nrrd_ext = '.nrrd'


def gen_heatmap_peak(filename):
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_dir, filename)

    heatmap, _ = nrrd.read(src_path)
    print(filename, heatmap.shape, heatmap.dtype)

    gauss = scipy.ndimage.filters.gaussian_filter(heatmap, 3, truncate=5)
    thresh = gauss > 0.2
    local_max = maximum_filter(gauss, size=6) == gauss
    filtered_max = np.logical_and(local_max, thresh)
    filtered_max = filtered_max.astype(np.int32)

    nrrd.write(dst_path, filtered_max)


src_list = os.listdir(args.srcdir)

if args.parallel:
    Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(
        delayed(gen_heatmap_peak)(filename) for filename in src_list)
else:
    for filename in src_list:
        gen_heatmap_peak(filename)
