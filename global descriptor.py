import numpy as np
import time
import os
import retrieve
import trimesh
from scipy.spatial.distance import cdist




def keyDistribution(points, b_num=100):
    center = np.sum(points, axis=0) / points.shape[0]
    dvector = points - center
    distances = np.linalg.norm(dvector, axis=1)
    distances = distances / np.max(distances)
    h = np.histogram(distances, bins=b_num, range=(0, 1))[0]
    return h / 10000


def sparseCoding(DLFS: np.ndarray):
    distances = cdist(DLFS, codebook)
    assignments = np.argmin(distances, axis=1)
    sparse_descriptor = np.zeros(num_clusters)
    for j in range(num_clusters):
        mask = assignments == j
        sparse_descriptor[j] = np.sum(mask)
    return sparse_descriptor


key_indices = np.load(r'C:\Users\Admin\CAD_parts/key_indices.npy', allow_pickle=True)
codebook = np.load('C:/Users/Admin/CAD_parts/codebook.npy')
num_clusters = codebook.shape[0]
DLFS_set = np.load('C:/Users/Admin/CAD_parts/con_DLFSs.npy', allow_pickle=True)

labels = np.load('C:/Users/Admin/CAD_parts/labels.npy')
weights = np.log(labels.shape[0] / np.bincount(labels))
weights = codebook.shape[0] * weights / weights.sum()

descriptors = np.zeros((DLFS_set.shape[0], num_clusters+100))

data_dir = r'C:/Users/Admin/CAD_parts'
k = 0
tic = time.time()

for category in os.listdir(data_dir):
    if not os.path.isdir(os.path.join(data_dir, category)):
        continue
    # if not category.endswith('mesh'):
    #    continue
    for filename in os.listdir(os.path.join(data_dir, category, 'STL')):
        if not filename.endswith('.stl'):
            continue
        path = os.path.join(data_dir, category, 'STL', filename)
        mesh = trimesh.load_mesh(path)
        points = trimesh.sample.sample_surface(mesh, 10000)[0]
        points = np.unique(points, axis=0)
        distribution = keyDistribution(points, b_num=100)

        descriptor = sparseCoding(DLFS_set[k])
        descriptors[k] = np.concatenate([distribution, descriptor])
        k += 1
    print(category, 'finished')
'''
for file in os.listdir(os.path.join(data_dir, '新建文件夹')):
    if not file.endswith('stl'):
        continue
    path = os.path.join(data_dir, '新建文件夹', file)
    mesh = trimesh.load_mesh(path)
    points = trimesh.sample.sample_surface(mesh, 10000)[0]
    distribution = keyDistribution(points, b_num=100)
    descriptor = sparseCoding(DLFS_set[k])
    descriptors[k] = np.concatenate([distribution, descriptor])
    k += 1
'''
print(time.time() - tic, 's')
np.save('C:/Users/Admin/CAD_parts/distribution_sparseDescriptors.npy', descriptors)
