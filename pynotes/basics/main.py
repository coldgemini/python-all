import argparse
from joblib import Parallel, delayed


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.parallel:
        Parallel(n_jobs=n_jobs, backend="multiprocessing")(
            delayed(getcrop)(basename) for basename in file_lines)
    else:
        for basename in sorted(file_lines):
            getcrop(basename)


if __name__ == '__main__':
    main()
