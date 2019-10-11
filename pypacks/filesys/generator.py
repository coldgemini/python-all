import os


def basename_file_and_full_paths(filelist, *dirs):
    for filename in filelist:
        yield [os.path.splitext(filename)[0], filename] + [os.path.join(dir_path, filename) for dir_path in dirs]


def main():
    dir_path = '/home/xiang/Downloads'
    filelist = os.listdir(dir_path)
    for [basename, filename, fpath1, fpath2] in basename_file_and_full_paths(filelist, 'path1', 'path2'):
        print(basename)
        print(filename)
        print(fpath1)
        print(fpath2)

    for [basename, filename] in basename_file_and_full_paths(filelist):
        print(basename)
        print(filename)


if __name__ == '__main__':
    main()
