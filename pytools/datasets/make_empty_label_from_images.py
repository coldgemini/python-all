import os
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--srcdir", type=str, help="src image folder")
parser.add_argument("-d", "--dstdir", type=str, help="dst image folder")
args = parser.parse_args()

src_dir = args.srcdir
dst_dir = args.dstdir


def gen_label(filename):
    print(filename)
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_dir, filename)

    # image = cv2.imread(src_path)
    image = Image.open(src_path)
    image_gray = image.convert("L")
    np_gray = np.asarray(image_gray)
    print("image_gray")
    print(np_gray.shape)
    print(np_gray.dtype)
    print(np.unique(np_gray))

    np_label = np.zeros_like(np_gray, dtype=np.uint8)
    label = Image.fromarray(np_label)
    label = label.convert("L")
    label.save(dst_path)

    print("label")
    print(np_label.shape)
    print(np_label.dtype)
    print(np.unique(np_label))


if __name__ == '__main__':

    src_list = os.listdir(args.srcdir)

    for filename in src_list:
        gen_label(filename)
