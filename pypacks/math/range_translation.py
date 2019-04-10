import numpy as np


def scale_range(data_in, clip_min, clip_max, out_min, out_max):
    data = data_in.astype(np.float64)
    data = np.clip(data, clip_min, clip_max)
    data = (data - clip_min) / (clip_max - clip_min)
    data = data * (out_max - out_min) + out_min
    return data.astype(data_in.dtype)
