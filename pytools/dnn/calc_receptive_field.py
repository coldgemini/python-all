# calculate receptive field

import numpy as np

cnn_sample = [{'f': 4, 's': 3}, {'f': 3, 's': 1}]


def rf(cnn):
    assert len(cnn) >= 1, "height of cnn layers must be >= 1"
    rf_list = [1]
    for i in range(len(cnn)):
        print(i)
        stride_i = int(np.prod([x['s'] for x in cnn[:i + 1]]))
        rf_i = rf_list[-1] + (cnn[i]['f'] - 1) * stride_i
        rf_list.append(rf_i)
        print(f"The receptive field of layer {i + 1} is {rf_i}")


if __name__ == '__main__':
    rf(cnn_sample)
