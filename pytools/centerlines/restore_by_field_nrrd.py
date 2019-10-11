import numpy as np
import os
import nrrd
from pypacks.ops3D.centerlines import restore_from_fields
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--srcmask", type=str, help="src image")
parser.add_argument("-d", "--dst", type=str, help="src image")  # transforming mask according to dst image size
parser.add_argument("--dstmask", type=str, help="dst image")
parser.add_argument("-f", "--field", type=str, help="displacement field dir")
parser.add_argument("-l", "--list", type=str, help="processing list")
parser.add_argument("-n", "--n_jobs", type=int, default=40, help="parallel jobs")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
args = parser.parse_args()
dstdir = args.dst
srcmask_dir = args.srcmask
dstmask_dir = args.dstmask
field_dir = args.field
listfile = args.list
n_jobs = args.n_jobs
parallel = args.parallel

# read listfile
listfile_h = open(listfile, "r")
file_lines = listfile_h.readlines()
file_lines = [line.rstrip() for line in file_lines]
listfile_h.close()


def restore_mask(filename):
    print(filename)

    dstpath = os.path.join(dstdir, filename)
    srcmask_path = os.path.join(srcmask_dir, filename)
    dstmask_path = os.path.join(dstmask_dir, filename)
    field_path = os.path.join(field_dir, filename)

    dstdata, _ = nrrd.read(dstpath)
    srcmask_data, _ = nrrd.read(srcmask_path)
    field_data, _ = nrrd.read(field_path)
    dstmask_data = np.zeros_like(dstdata, dtype=np.uint8)

    restore_from_fields(dstmask_data, field_data, srcmask_data)

    nrrd.write(dstmask_path, dstmask_data)

    return


if parallel:
    Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(restore_mask)(filename) for filename in file_lines)
else:
    for filename in file_lines:
        restore_mask(filename)
