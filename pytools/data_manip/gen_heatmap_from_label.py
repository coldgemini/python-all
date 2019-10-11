import os
import nrrd
from image_utils.heatmap import gen_heatmap_3d_from_label
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src image dir")
parser.add_argument("-d", "--dst", type=str, help="dst slice dir")
parser.add_argument("-n", "--n_jobs", type=int, default=2, help="parallel jobs")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
args = parser.parse_args()

src_dir = args.src
dst_dir = args.dst

src_list = os.listdir(src_dir)


def gen_heatmap_from_label(filename):
    print(filename)
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_dir, filename)

    label, _ = nrrd.read(src_path)
    heatmap = gen_heatmap_3d_from_label(label, sigma=4)

    nrrd.write(dst_path, heatmap)


if args.parallel:
    Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(
        delayed(gen_heatmap_from_label)(filename) for filename in src_list)
else:
    for filename in src_list:
        gen_heatmap_from_label(filename)
