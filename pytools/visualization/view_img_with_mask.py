import cv2
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="dst image dir")
parser.add_argument("-m", "--mask", type=str, help="dst image dir")
parser.add_argument("--size", type=int, default=512, help="dst image dir")
args = parser.parse_args()
srcdir = args.src
mask_dir = args.mask
size = args.size

filelist = os.listdir(srcdir)

for filename in filelist:
    imgpath = os.path.join(srcdir, filename)
    mskpath = os.path.join(mask_dir, filename)
    if not os.path.isfile(mskpath):
        continue
    print(filename)
    img = cv2.imread(imgpath)
    msk = cv2.imread(mskpath)
    print(np.unique(msk))
    msk[:, :, 0] = 0
    msk[:, :, 1] = 0
    msk = msk // 6
    print(np.unique(msk))
    masked_img = img + msk

    img = cv2.resize(img, (size, size))
    # msk = cv2.resize(msk, (size, size))
    masked_img = cv2.resize(masked_img, (size, size))
    # merge = np.concatenate((img, msk), axis=1)
    merge = np.concatenate((img, masked_img), axis=1)
    cv2.imshow('merge', merge)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break
