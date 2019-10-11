import cv2
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="dst image dir")
parser.add_argument("-m", "--msk", type=str, help="dst image dir")
parser.add_argument("-n", "--filename", type=str, default=None, help="dst image dir")
parser.add_argument("--size", type=int, default=512, help="dst image dir")
parser.add_argument("-o", "--sort", action='store_true', default=False, help="if sort")
args = parser.parse_args()

filelist = os.listdir(args.src)

if args.sort:
    filelist = sorted(filelist)

if args.filename is not None:
    filelist = [args.filename]

key = 0
for filename in filelist:
    print(filename)
    npzpath = os.path.join(args.src, filename)
    npzmskpath = os.path.join(args.msk, filename)
    npz = np.load(npzpath)
    nparr = npz['arr_0']
    npzmsk = np.load(npzmskpath)
    npmskarr = npzmsk['arr_0']

    if len(nparr.shape) == 2:
        pass
        img = nparr
    else:
        _, _, zS = nparr.shape
        idx = zS // 2
        img = nparr[:, :, idx]

    img = cv2.resize(img, (args.size, args.size))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    msk = cv2.resize(npmskarr, (args.size, args.size))
    msk = cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)
    msk[:, :, 1:3] = 0
    msk = msk * 33
    overlay = img + msk

    merge = np.concatenate((img, overlay), axis=1)

    cv2.imshow('merge', merge)
    while True:
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
