import cc3d
import numpy as np

# labels_in = np.ones((3, 3, 3), dtype=np.int32)
# labels_in = np.zeros((4, 4, 4), dtype=np.uint8)
# labels_in = np.eye(3, dtype=np.uint8)
labels_in = np.zeros((3, 3), dtype=np.uint8)
labels_in[0, 0] = 1
# labels_in[1, 1] = 2
labels_in[2, 2] = 1
# labels_in[3, 3, 3] = 3
print("in")
print(labels_in)
labels_out = cc3d.connected_components(labels_in, max_labels=3)
print("out")
print(labels_out)

# You can extract individual components like so:
N = np.max(labels_out)
print("N", N)
for segid in range(1, N + 1):
    print("haha")
    extracted_image = labels_out * (labels_out == segid)
    print(extracted_image)
    # process(extracted_image)

# We also include a 26-connected region adjacency graph function
# that returns a set of undirected edges. It is not optimized
# (100x slower than connected_components) but it could be improved.
# graph = cc3d.region_graph(labels_out)
