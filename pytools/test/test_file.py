import os

cur_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(cur_dir, '..'))
print(cur_dir)
print(parent_dir)
