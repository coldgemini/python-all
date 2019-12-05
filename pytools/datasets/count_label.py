import os
import pickle
import nrrd
import numpy as np
import argparse
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--srcdir", type=str, help="src image folder")
parser.add_argument("-n", "--njobs", type=int, default=1, help="num of shards")
parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
args = parser.parse_args()

src_dir = args.srcdir


def count_label(filename):
    print(filename)
    src_path = os.path.join(src_dir, filename)

    label, _ = nrrd.read(src_path)
    count = np.count_nonzero(label == 1)
    print(f"count: {count}")
    return count


if __name__ == '__main__':

    src_list = os.listdir(args.srcdir)
    if args.parallel:
        count_list = Parallel(n_jobs=args.njobs, backend="multiprocessing")(
            delayed(count_label)(filename) for filename in src_list)

        pkl_output = open('count_list.pkl', 'wb')
        pickle.dump(count_list, pkl_output)
        pkl_output.close()
    else:
        for filename in src_list:
            count_label(filename)
