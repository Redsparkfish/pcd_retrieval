import numpy as np
import time
import os
from scipy.spatial.distance import cdist


def sparseCoding(CSet):
    distances = cdist(CSet, centers, 'cosine')
    assignments = np.argmin(distances, axis=1)
    SBoWC_descriptor = np.zeros(num_clusters)
    for j in range(num_clusters):
        mask = assignments == j
        SBoWC_descriptor[j] = np.sum(mask)
    return SBoWC_descriptor / np.sum(SBoWC_descriptor)


centers = np.load('clique_bag.npy')
names = np.load('names.npy')
num_clusters = centers.shape[0]
CSet_collection = np.load('CSet_collection.npy', allow_pickle=True)

descriptors = np.zeros((len(names), centers.shape[0]))

data_dir = 'c:/users/admin/modelnet40_normal_resampled'
train_category = np.loadtxt('c:/users/admin/modelnet40_normal_resampled/modelnet10_shape_names.txt', dtype=str)
k = 0
tic = time.time()

for category in train_category:
    n = 0
    for filename in os.listdir(os.path.join(data_dir, category)):
        if not filename.endswith('.txt'):
            continue
        if not n < 100:
            break
        path = os.path.join(data_dir, category, filename)
        descriptor = sparseCoding(CSet_collection[k])
        descriptors[k] = descriptor
        k += 1
        n += 1
    print(category, 'finished')

print(time.time() - tic, 's')
np.save('SBoWC_descriptors', descriptors)
