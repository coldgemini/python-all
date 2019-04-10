import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', nargs='+', type=int)
args = parser.parse_args()
my_tuple = tuple(args.data)