import time
import os

import numpy as np
import trimesh.sample
from scipy.spatial.distance import cdist
from DLFS_calculation import *
from global_descriptor import D2, sparseCoding
import matplotlib.pyplot as plt


def calcDescriptors(mesh, codebook):
    points = trimesh.sample.sample_surface(mesh, 20000)[0]
    mr = meshResolution(points)
    key_indices, neighbor_indices, eigvectors = computeISS(points, rate=1, radius=2.5 * mr)
    LMA = getLMA(points, eigvectors, radius=7 * mr)
    DLFSs = getDLFS(points, LMA, key_indices, R=20 * mr)
    DLFSs = DLFSs.reshape(DLFSs.shape[0], DLFSs.shape[1]*DLFSs.shape[2])
    distribution = D2(points)
    bof_descriptor = sparseCoding(DLFSs, codebook)
    return distribution/20000, bof_descriptor


def retrieve_test(mesh, k, codebook):
    query_bof = calcDescriptors(mesh, codebook)[1]
    distances = cdist(query_bof.reshape(1, query_bof.shape[0]), descriptor_set, metric='canberra')
    idx = np.argsort(distances[0])[:k+1]
    closest = names[idx]
    return closest


def pr_curve(retrieval_results, category_name, category_total, resolution=0.1):
    k = 0
    precision = []
    recall = []
    n = len(retrieval_results)
    for i, result in enumerate(retrieval_results):
        if result.startswith(category_name):
            k += 1
        precision.append(k / (i+1))
        recall.append(k / category_total)
    return np.array(precision), np.array(recall)


descriptor_set = np.load('C:/Users/Admin/CAD_parts/descriptors.npy')
names = np.load('C:/Users/Admin/CAD_parts/names.npy')
codebook = np.load(r'C:/Users/Admin/CAD_parts/codebook.npy')
# data_dir = r'C:/Users/Admin/CAD_parts'
# category = 'Bearings'
# n_category = 58
# n_neighbor = 20
# precision = np.zeros((n_category, n_neighbor))
# recall = np.zeros((n_category, n_neighbor))
# j = 0
# tic = time.time()
# for file in os.listdir(os.path.join(data_dir, category, 'STL')):
#     if not file.endswith('stl'):
#         continue
#     print(file)
#     index = list(names).index(file)
#     mesh = trimesh.load_mesh(os.path.join(data_dir, category, 'STL', file))
#     results = retrieve_test(mesh, n_neighbor, codebook)[:n_neighbor]
#     query_precision, query_recall = pr_curve(results, category, n_category)
#     precision[j] = query_precision
#     recall[j] = query_recall
#     j += 1
# print(time.time() - tic)
# avg_p = np.average(precision, axis=0)
# avg_r = np.average(recall, axis=0)
# plt.plot(avg_p)
# plt.plot(avg_r)
# plt.show()



def retrieve(query_name, k):
    d2_list = np.load('C:/Users/Admin/CAD_parts/d2_list.npy')
    size = d2_list.shape[0]
    n = 400
    nn = NearestNeighbors(n_neighbors=n+1, metric='canberra')
    nn.fit(d2_list)
    distances, indices_list = nn.kneighbors(d2_list)

    names = np.load('C:/Users/Admin/CAD_parts/names.npy')

    # index = (names == query_name)
    # distances = cdist(descriptor_set[index], descriptor_set[indices_list[index]][0], metric='canberra')
    # idx = np.argsort(distances[0])[:k+1]
    # closest = names[indices_list[index][0][idx]]


    summary = 0
    
    for i in range(size):
        str_list = names[i].split('_')
        n = len(str_list)
        category = ''
        for j in range(n-1):
            category += str_list[j] + '_'
        name = names[i]
        # print(category)
        # print(name)
        distances = cdist(descriptor_set[i].reshape(1, descriptor_set[i].shape[0]), descriptor_set[indices_list[i]], metric='canberra')
        idx = np.argsort(distances[0])[:k+1]
        closest = names[indices_list[i][idx[:k + 1]]]
        if category == '':
            print(name)
            print(closest)
        print(name, ':')
        for j in range(k+1):
            if closest[j].startswith(category) and closest[j] != name:
                print(closest[j])
                summary += 1
            #elif closest[j] != name:
            #    print(name,  closest[j], '\n')
        print('')
    
    print(summary/(k * size))

    return

retrieve('a', k=10)

'''
nn = NearestNeighbors(n_neighbors=k+1, metric='canberra')
nn.fit(descriptors)
distances, indices_list = nn.kneighbors(descriptors)

summary = 0
for i in range(size):
    str_list = names[i].split('_')
    n = len(str_list)
    category = ''
    for j in range(n - 1):
        category += str_list[j] + '_'
    name = names[i]
    closest = names[indices_list[i]][1:]
    for j in range(k):
        if closest[j].startswith(category):
            summary += 1
print(summary/(k * size))
'''
'''
input_name = 'Bearings_24161c0b-afc7-449c-a803-62ae4e88cc73.stl'
ft = 58
result = retrieve(input_name, 2*ft+1)
summary = 0
category = 'Bearings'
for j in range(2*ft+1):
    if result[j].startswith(category):
        summary += 1

print(summary/ft)
'''