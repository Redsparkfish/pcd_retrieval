from scipy.spatial.distance import cdist
import numpy as np
import trimesh
import os


def D2(points: np.ndarray, b_num=100):
    size = points.shape[0]
    permutation = np.random.permutation(size)
    dvector = points - points[permutation]
    distances = np.linalg.norm(dvector, axis=1)
    distances = distances
    return np.histogram(distances, bins=b_num)[0]

def sparseCoding(DLFS: np.ndarray, codebook):
    distances = cdist(DLFS, codebook)
    num_clusters = codebook.shape[0]
    assignments = np.argmin(distances, axis=1)
    sparse_descriptor = np.zeros(num_clusters)
    for j in range(num_clusters):
        mask = assignments == j
        sparse_descriptor[j] = np.sum(mask)
    return sparse_descriptor


def computeGlobalDescriptors(data_dir, DLFS_set, codebook):
    num_clusters = codebook.shape[0]
    descriptors = np.zeros((DLFS_set.shape[0], num_clusters))
    D2s = np.zeros((DLFS_set.shape[0], 100))
    k = 0
    names = []
    categories = []
    for category in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, category)):
            continue
        if category.lower() in ['update', 'temp_update']:
            continue
        for filename in os.listdir(os.path.join(data_dir, category, 'PCD')):
            if not filename.endswith('.npy'):
                continue
            names.append(filename[:-4]+'.stl')
            categories.append(category)
            path = os.path.join(data_dir, category, 'PCD', filename)
            points = np.load(path)
            distribution = D2(points)
            D2s[k] = distribution
            descriptor = sparseCoding(DLFS_set[k], codebook)
            descriptors[k] = descriptor
            k += 1
        print(category, 'finished')
    return descriptors, D2s, names, categories
