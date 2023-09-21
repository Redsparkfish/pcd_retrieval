import json
import os
import numpy as np
import shutil
from sklearn.neighbors import NearestNeighbors
from DLFS_calculation import meshResolution, getNeighbors, getKNeighbors
from mlpack import knn
import time


data = np.random.randn(50000, 3)
mr = meshResolution(data)

tic = time.time()
k_neighbor_indices = getKNeighbors(data, 400)
tic1 = time.time()
print(tic1 - tic)

print(k_neighbor_indices[0].shape)

index = 0
distances = [np.linalg.norm(data[k_neighbor_indices[index][i]]-data[index]) for i in range(k_neighbor_indices[index].shape[0])]
distances = np.array(distances)
print(distances)
print(distances/mr)

# neighbor_indices = getNeighbors(data, radius=20*mr)
# tic2 = time.time()
# print(tic2 - tic1)
