import numpy as np
import pickle
# import pprint
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, help="src image folder")
args = parser.parse_args()

# pkl_output = open('count_list.pkl', 'wb')
# pickle.dump(count_list, pkl_output)
# pkl_output.close()


pkl_file = open('count_list.pkl', 'rb')

count_list = pickle.load(pkl_file)
# pprint.pprint(count_list)
pkl_file.close()

list0 = [num for num in count_list if num == 0]
list10 = [num for num in count_list if num < 10]
list100 = [num for num in count_list if num < 100]
list1000 = [num for num in count_list if num < 1000]
list5000 = [num for num in count_list if num < 5000]
print(f"0: {len(list0)}")
print(f"10: {len(list10)}")
print(f"100: {len(list100)}")
print(f"1000: {len(list1000)}")
print(f"5000: {len(list5000)}")
print(f"all: {len(count_list)}")
# plt.hist(count_list, bins='auto')
# plt.hist(count_list, bins=np.arange(0, 40000, 100))
# plt.hist(count_list, bins=np.arange(0, 200, 5))
# plt.show()
