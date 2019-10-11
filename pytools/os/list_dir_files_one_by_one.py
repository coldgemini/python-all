import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src file dir")
args = parser.parse_args()

dir_path = os.path.abspath(args.src)
filelist = os.listdir(dir_path)

for filename in sorted(filelist):
    full_path = os.path.join(dir_path, filename)
    print(full_path)
    # input("go next?")
    input("press Enter to go next")
