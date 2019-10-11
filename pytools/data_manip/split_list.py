import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--list", type=str, default=None, help="list file")
parser.add_argument("-s", "--splits", nargs='+', type=str, default=("train", "val"), help="splits")
args = parser.parse_args()

abspath = os.path.abspath(args.list)
folder = os.path.dirname(abspath)
basename = os.path.basename(abspath)
basename_noext, ext = os.path.splitext(basename)
train_file_path = os.path.join(folder, basename_noext + '_train' + ext)
val_file_path = os.path.join(folder, basename_noext + '_val' + ext)

# print(abspath)
# read listfile
input_res_file = open(abspath, "r")
file_lines = input_res_file.readlines()
file_lines = [line.rstrip() for line in file_lines]
input_res_file.close()


random.shuffle(file_lines)

size = len(file_lines)
num_val = size // 10
num_train = size - num_val

train_file_lines = file_lines[0:num_train]
val_file_lines = file_lines[-num_val:]

# write train.txt file
out_text_file = open(train_file_path, "w")
out_text_file.write('\n'.join(train_file_lines))
out_text_file.close()
# write val.txt file
out_text_file = open(val_file_path, "w")
out_text_file.write('\n'.join(val_file_lines))
out_text_file.close()
