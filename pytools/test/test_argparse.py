import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="src image dir")
args = parser.parse_args()
srcdir = args.src

print(srcdir)
print(None)