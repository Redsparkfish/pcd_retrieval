import time
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist



def getKNeighbors(points, k):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(points)
    distances, k_neighbor_indices = knn.kneighbors()
    return k_neighbor_indices


def spatialGraph(points, key_indices, k=12):
    keypoints = points[key_indices]
    size = len(keypoints)
    kneighbors = getKNeighbors(keypoints, k)
    graph = np.zeros((size, size))
    for i in range(size):
        graph[i, i] = 1
        for j in kneighbors[i]:
            graph[i, j] = 1
            graph[j, i] = 1
    return graph


def classify(features, codebook):
    distances = cdist(features, codebook)
    features_category = np.argmin(distances, axis=1)
    return features_category


def correlation(graph, feature_category, threshold=6):
    size = codebook.shape[0]
    CMat = np.zeros((size, size))
    for i in range(size):
        CMat[i, i] = threshold + 1
        for j in range(i+1, size):
            CMat[i, j] = np.count_nonzero(graph[feature_category == i][:, feature_category == j])
            if CMat[i, j] < threshold:
                CMat[i, j] = 0
            CMat[j, i] = CMat[i, j]
    return CMat


def regraph(graph: np.ndarray, feature_category, CMat):
    nonzero = graph.nonzero()
    for i in range(len(nonzero[0])):
        if CMat[feature_category[nonzero[0][i]], feature_category[nonzero[1][i]]] == 0:
            graph[nonzero[0][i], nonzero[1][i]] = 0
    return graph


def getCSet(graph: np.ndarray, feature_category):
    size = len(graph)
    marked = np.zeros(size)
    mark = 1
    for i in range(size):
        num_neighbors = np.count_nonzero(graph[i])
        if num_neighbors > 0:
            if marked[i] == 0:
                marked[i] = mark
                mark += 1
            for j in graph[i].nonzero()[0]:
                if marked[j] == 0:
                    marked[j] = marked[i]
                    graph[j, i] = 0
        else:
            if marked[i] == 0:
                marked[i] = mark
                mark += 1

    CSet = np.zeros((mark-1, codebook.shape[0]))
    for i in range(1, mark):
        clique = np.zeros(codebook.shape[0])
        entry_of_mark_i = marked == i
        for category in feature_category[entry_of_mark_i]:
            clique[category] += 1
        CSet[i-1] = clique
    return CSet


key_indices = np.load('key_indices.npy', allow_pickle=True)
DLFS_set = np.load('con_DLFSs.npy', allow_pickle=True)
codebook = np.load('codebook.npy')
names = np.load('names.npy')
data_dir = 'c:/users/admin/modelnet40_normal_resampled'
training_category = np.loadtxt('c:/users/admin/modelnet40_normal_resampled/modelnet10_shape_names.txt', dtype=str)

n = 0
CSet_collection = []
tic = time.time()
for category in training_category:
    if not os.path.isdir(os.path.join(data_dir, category)):
        continue
    k = 0
    for filename in os.listdir(os.path.join(data_dir, category)):
        if not k < 100:
            break
        if not filename.endswith('.txt'):
            continue
        points = np.loadtxt(os.path.join(data_dir, category, filename), delimiter=',')
        points = points[:, :3]
        points = np.unique(points, axis=0)
        graph = spatialGraph(points, key_indices[n])
        features_category = classify(DLFS_set[n], codebook)
        CMat = correlation(graph, features_category)
        graph = regraph(graph, features_category, CMat)
        CSet = getCSet(graph, features_category)
        CSet_collection.append(CSet)
        k += 1
        n += 1
    print(category)
print(time.time() - tic)

arr = np.empty(len(CSet_collection), object)
arr[:] = CSet_collection
np.save('CSet_collection.npy', arr)
