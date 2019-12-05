import argparse
from joblib import Parallel, delayed

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument("-n", "--n_jobs", type=int, default=3, help="parallel jobs")
    parser.add_argument("-p", "--parallel", action='store_true', default=False, help="if parallel")
    args = parser.parse_args()
    return args


def myfunc(data):
    pass


def main():
    global args
    args = parse_args()

    if args.parallel:
        Parallel(n_jobs=n_jobs, backend="multiprocessing")(
            delayed(myfunc)(basename) for basename in file_lines)
    else:
        for basename in sorted(file_lines):
            myfunc(basename)


if __name__ == '__main__':
    main()
