import cv2
import numpy as np
import os
# from pypacks.math.range_translation import scale_range
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="dst image dir")
parser.add_argument("--size", type=int, default=512, help="dst image dir")
args = parser.parse_args()
srcdir = args.src
size = args.size

filelist = os.listdir(srcdir)

key = 0
for filename in filelist:
    npzpath = os.path.join(srcdir, filename)
    print(filename)
    npz = np.load(npzpath)
    print(npz.files)
    nparr = npz['arr_0']
    # nparr = scale_range(nparr, -1000, 1000, 1, 254)
    print(nparr.shape)
    print(np.unique(nparr))
    if len(nparr.shape) == 3:
        _, _, zS = nparr.shape
        for idx in range(zS):
            img = nparr[:, :, idx]
            img = cv2.resize(img, (size, size))
            cv2.imshow('merge', img)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
    else:
        img = nparr
        img = cv2.resize(img, (size, size))
        cv2.imshow('merge', img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    if key == ord('q'):
        break
