import os
import nrrd
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src image dir")
parser.add_argument("-n", "--n_jobs", type=int, default=2, help="parallel jobs")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
parser.add_argument("-v", "--verbose", action='store_true', default=False, help="verbose")
args = parser.parse_args()


def validate(filename):
    if args.verbose:
        print(filename)
    srcpath = os.path.join(args.src, filename)
    try:
        srcdata, _ = nrrd.read(srcpath)
    except Exception as e:
        print(str(e))
        print("error: ", srcpath)


filelist = os.listdir(args.src)

if args.parallel:
    Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(
        # Parallel(n_jobs=args.n_jobs, require='sharedmem')(
        delayed(validate)(filename) for filename in filelist)
else:
    for filename in filelist:
        validate(filename)
