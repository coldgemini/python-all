import os
import cv2
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--srcdir", type=str, help="src folder")
parser.add_argument("-d", "--dstdir", type=str, help="dst folder")
parser.add_argument("-hh", "--height", type=int, default=256, help="height")
parser.add_argument("-ww", "--width", type=int, default=256, help="width")
parser.add_argument("-i", "--interp", type=str, default='nearest', choices=['nearest', 'linear', 'cubic'],
                    help="interpolation")
parser.add_argument("-c", "--cores", type=int, default=32, help="njobs")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parellel")
args = parser.parse_args()
srcdir = args.srcdir
dstdir = args.dstdir
height = args.height
width = args.width
interp = args.interp
n_jobs = args.cores
parallel = args.parallel

filelist = os.listdir(srcdir)


def resize_batch(filename):
    # for filename in filelist:
    print(filename)
    srcpath = os.path.join(srcdir, filename)
    dstpath = os.path.join(dstdir, filename)
    img = cv2.imread(srcpath)
    if interp == 'linear':
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    elif interp == 'cubic':
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    else:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(dstpath, img)


if parallel:
    Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(resize_batch)(filename) for filename in filelist)
else:
    for filename in filelist:
        resize_batch(filename)
