import nrrd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src image dir")
parser.add_argument("-d", "--dst", type=str, help="dst slice dir")
args = parser.parse_args()

src_path = args.src
dst_path = args.dst

src_data, _ = nrrd.read(src_path)

clip_data = src_data[50 - 32:50 + 32, 50 - 32:50 + 32, 230 - 32:230 + 32]
print(clip_data.dtype)
print(clip_data.shape)

nrrd.write(dst_path, clip_data)
