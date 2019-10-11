import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src file dir")
parser.add_argument("-m", "--mask", type=str, help="src file dir")
args = parser.parse_args()

dir_path = os.path.abspath(args.src)
maskdir_path = os.path.abspath(args.mask)
filelist = os.listdir(dir_path)
num_files = len(filelist)

for idx, filename in enumerate(sorted(filelist)):
    full_path = os.path.join(dir_path, filename)
    mask_full_path = os.path.join(maskdir_path, filename)
    print("data: {}".format(full_path))
    print("mask: {}".format(mask_full_path))
    print("idx: {0} in total {1}".format(idx, num_files))
    # input("go next?")
    input("press Enter to go next")
