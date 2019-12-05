"""
+word
#
&
@
$1
"""

import os

out_dir = '/home/xiang/Shared'
filepath = '/home/xiang/Shared/word_full.txt'
num_part = 2500

# read listfile
file_h = open(filepath, "r")
file_lines = file_h.readlines()
file_lines = [line.strip() for line in file_lines]
file_h.close()
full_word_list = file_lines

print(file_lines[0:10])
print(len(file_lines))

total_length = len(file_lines)

partitions = total_length // num_part + 1

for idx in range(partitions):
    id_start = idx * num_part
    print(id_start)
    out_filename = str(id_start) + '-' + str(id_start + num_part) + '.txt'
    out_file_path = os.path.join(out_dir, out_filename)
    word_list = full_word_list[id_start:id_start + num_part]
    nested = [[f"+{word}", "#", "&", "@", "$1"] for word in word_list]
    flattened = [val for sublist in nested for val in sublist]
    # print(flattened)

    # write train.txt file
    out_file_h = open(out_file_path, "w")
    out_file_h.write('\n'.join(flattened))
    out_file_h.close()
