# read listfile
input_res_file = open(filelist, "r")
file_lines = input_res_file.readlines()
file_lines = [line.rstrip() for line in file_lines]
input_res_file.close()

from glob import glob
pair_list = glob('{}/*.nii.gz'.format(self.traindata_dir))

import os
for path, subdirs, files in os.walk(niftifolder):
    for name in files:
        niftipath = os.path.join(path, name)

filelist = os.listdir(dir)


# write train.txt file
out_text_file = open(train_list_file, "w")
out_text_file.write('\n'.join(train_filelist))
out_text_file.close()

>>> glob.glob1("some_dir", "*.png")
['foo.png', 'bar.png', ...]

>>> glob.glob("some_dir/*.png")
['/home/michael/A_dir/B_dir/some_dir/foo.png',
'/home/michael/A_dir/B_dir/some_dir/bar.png',
...]