from DLFS_calculation import computeDLFS
from bag_of_feature import construct_bof, construct_codebook
from global_descriptor import computeGlobalDescriptors
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import os
import numpy as np

data_dir = input('Enter the path of database: ')
'''
DLFS_features, key_indices, names = computeDLFS(data_dir)
bof, con_DLFS = construct_bof(DLFS_features)
codebook, labels = construct_codebook(bof)
descriptors, D2s = computeGlobalDescriptors(data_dir, con_DLFS, codebook)
np.save(os.path.join(data_dir, 'descriptors.npy'), descriptors)
np.save(os.path.join(data_dir, 'D2s.npy'), D2s)
np.save(os.path.join(data_dir, 'names.npy'), names)
np.save(os.path.join(data_dir, 'bof.npy'), bof)
np.save(os.path.join(data_dir, 'codebook.npy'), codebook)
'''

con_DLFS = np.load(os.path.join(data_dir, 'con_DLFSs.npy'))
codebook = np.load(os.path.join(data_dir, 'codebook.npy'))
descriptors = np.load(os.path.join(data_dir, 'descriptors.npy'))
D2s = np.load(os.path.join(data_dir, 'D2s.npy'))
names = np.load(os.path.join(data_dir, 'names.npy'))


p_descriptors = np.load(os.path.join(data_dir, 'parameter_descriptors.npy'))
pre_descriptors = np.hstack((D2s, p_descriptors))

size = p_descriptors.shape[0]
n = 500
nn = NearestNeighbors(n_neighbors=n+1, metric='canberra')
nn.fit(pre_descriptors)
distances, indices = nn.kneighbors(pre_descriptors)

k = 10

summary = 0
for i in range(size):
    category = '_'.join(names[i].split('_')[:-1])
    name = names[i]
    print(name)
    distances = cdist(descriptors[i].reshape(1, descriptors[i].shape[0]), descriptors[indices[i]], metric='canberra')
    idx = np.argpartition(distances[0], k+1)
    closest = names[indices[i][idx[:k+1]]]
    print(closest)

    for j in range(k+1):
        if closest[j] == name:
            continue
        elif closest[j].startswith(category):
            summary += 1
        elif category == 'Rotary_Shaft' or category == 'Keyway_Shaft':
            if closest[j].startswith('Rotary_Shaft') or closest[j].startswith('Keyway_Shaft'):
                summary += 1

    print('\n')

print(summary/(k * size))

nn = NearestNeighbors(n_neighbors=k+1, metric='canberra')
nn.fit(descriptors)
distances, indices = nn.kneighbors(descriptors)

summary = 0
for i in range(size):
    category = '_'.join(names[i].split('_')[:-1])
    name = names[i]
    closest = names[indices[i]]
    for j in range(k+1):
        if closest[j] == name:
            continue
        if closest[j].startswith(category):
            summary += 1
        elif category == 'Rotary_Shaft' or category == 'Keyway_Shaft':
            if closest[j].startswith('Rotary_Shaft') or closest[j].startswith('Keyway_Shaft'):
                summary += 1
print(summary/(k * size))
