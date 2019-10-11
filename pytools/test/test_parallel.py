# import cv2
import numpy as np
import os
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src image dir")
parser.add_argument("-n", "--n_jobs", type=int, default=3, help="parallel jobs")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
args = parser.parse_args()
srcdir = args.src
n_jobs = args.n_jobs
parallel = args.parallel

tmppath = 'tmp'
tmpArr = np.array([1,2,3])

filelist = os.listdir(srcdir)


def parafunc(filename):
    print(filename)
    print(tmppath)
    print(tmpArr.shape)


if parallel:
    print("parallel!!!")
    Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(parafunc)(pprocItem) for pprocItem in filelist)
else:
    for pprocItem in filelist:
        parafunc(pprocItem)
