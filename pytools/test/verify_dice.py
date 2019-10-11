import numpy as np
import sklearn.preprocessing
import nrrd

# label_path = '/data2/home/zhouxiangyong/Workspace/Dev/AortaSlice/data/aorta_extract_mask1_aorta_vessel_fxc/1.2.840.113619.2.55.3.2831164355.488.1510306320.268.4.nrrd'
label_path = '/home/xiang/Tmp/1.2.840.113619.2.55.3.2831164355.488.1510306320.268.4.nrrd'

label, _ = nrrd.read(label_path)
shape = label.shape
print(shape)
classes = np.max(label) + 1
shape_onehot = shape + (classes,)
print(shape_onehot)
label_flat = label.flatten()
label_binarizer = sklearn.preprocessing.LabelBinarizer()
label_binarizer.fit(range(np.max(label) + 1))
b_f = label_binarizer.transform(label_flat)
b = b_f.reshape(shape_onehot)
print('b shape: {0}'.format(b.shape))
b = np.expand_dims(b, axis=0)
print('b shape: {0}'.format(b.shape))

dice = 0
for i in range(classes):
    # inse = np.sum(b[:, :, :, :, i] * b[:, :, :, :, i])
    inse = np.mean(b[:, :, :, :, i] * b[:, :, :, :, i])
    l = np.mean(b[:, :, :, :, i] * b[:, :, :, :, i])
    # l = np.sum(b[:, :, :, :, i])
    # l = np.mean(b[:, :, :, :, i])
    r = np.mean(b[:, :, :, :, i] * b[:, :, :, :, i])
    # r = np.sum(b[:, :, :, :, i])
    # r = np.mean(b[:, :, :, :, i])
    dice_per_class = (2 * inse) / (l + r)
    print("dice per class: {}".format(dice_per_class))
    dice = dice + dice_per_class
    print("dice step: {}".format(dice))

print("dice: {}".format(dice))
# a = [1, 0, 3]
# a = np.array([1, 0, 3])
# a = np.array([[1, 0, 3], [1, 0, 3]])
# print(a)
# print(a.shape)
# a = a.flatten()
# # a = label
# label_binarizer = sklearn.preprocessing.LabelBinarizer()
# label_binarizer.fit(range(np.max(a) + 1))
# b = label_binarizer.transform(a)
# print('{0}'.format(b))
# b = b.reshape(2, 3, 4)
# print('{0}'.format(b))
# print(b.shape)
