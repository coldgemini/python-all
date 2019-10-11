import numpy as np
import os
import nrrd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src image dir")
args = parser.parse_args()

src_dir = args.src
src_list = os.listdir(src_dir)


def print_label_percentage(label):
    np_shape = np.array(label.shape, np.int32)
    np_shape = np_shape.astype(np.float64)
    total = np.prod(np_shape)
    l0 = np.where(label == 0, 1, 0)
    l0_sum = np.sum(l0)
    l1 = np.where(label == 1, 1, 0)
    l1_sum = np.sum(l1)
    l2 = np.where(label == 2, 1, 0)
    l2_sum = np.sum(l2)
    l3 = np.where(label == 3, 1, 0)
    l3_sum = np.sum(l3)
    l4 = np.where(label == 4, 1, 0)
    l4_sum = np.sum(l4)
    perc = (l0_sum / total, l1_sum / total, l2_sum / total, l3_sum / total, l4_sum / total)
    return perc


for filename in src_list:
    print(filename)
    file_path = os.path.join(src_dir, filename)
    label, _ = nrrd.read(file_path)
    perc = print_label_percentage(label)
    print(perc)
