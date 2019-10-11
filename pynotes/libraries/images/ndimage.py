from scipy.ndimage import zoom

nparr = zoom(nparr, (oy / iy, ox / ix, oz / iz))

roi_aortacrop = zoom(roi_aortacrop, (xwidth / rst_shape[0], ywidth / rst_shape[1], zwidth / rst_shape[2]),
                     order=0, mode='nearest')
