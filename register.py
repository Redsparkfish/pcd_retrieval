import numpy as np
import pymeshlab
from scipy.spatial.distance import cdist
from DLFS_calculation import *


def poisson_sample(path, radius=pymeshlab.Percentage(1)):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    ms.apply_filter('generate_sampling_poisson_disk', radius=radius)
    points = ms.current_mesh().vertex_matrix()
    return points


def computeISS(points, t1=0.9, t2=0.9, radius=7 / 3):
    point_size = points.shape[0]
    index_set = np.arange(point_size)
    neighborSet = getNeighbors(points, radius=radius)
    k_neighborSet = getKNeighbors(points, 50)

    def calcWeight(index):
        weight = 1 / (neighborSet[index[0]].shape[0] + 1)
        return weight

    weights = np.apply_along_axis(calcWeight, -1, index_set.reshape(point_size, 1))

    def calcEig(index):
        i = index[0]
        if len(neighborSet[i]) < 50:
            neighborSet[i] = k_neighborSet[i]
        if i in neighborSet[i]:
            print(i)
        d_vectors = points[i] - points[neighborSet[i]]
        X = np.multiply(weights[neighborSet[i]].reshape(d_vectors.shape[0], 1), d_vectors).T
        Y = d_vectors
        cov = np.dot(X, Y) * weights[i]
        eigvalue, eigvector = np.linalg.eig(cov)
        idx = np.argsort(eigvalue)
        eigvalue = eigvalue[idx]
        eigvector = eigvector[idx]
        return np.vstack((eigvalue, eigvector))

    Eig = np.apply_along_axis(calcEig, -1, index_set.reshape(point_size, 1))
    eigvalues = Eig[:, 0, :]
    eigvectors = Eig[:, 1, :]

    def acceptIndex(index):
        i = index[0]
        if t1 * eigvalues[i][1] > eigvalues[i][0] and t2 * eigvalues[i][2] > eigvalues[i][1]:
            return True
        return False
    indices = index_set[np.apply_along_axis(acceptIndex, -1, index_set.reshape(point_size, 1))]

    unvisited = set(indices)
    key_indices = np.zeros(1, dtype=int)

    # 去掉某些太靠近的关键点
    while len(unvisited):
        core = list(unvisited)[np.random.randint(0, len(unvisited))]  # 从 关键点集T 中随机选取一个 关键点core
        core_neighbors = neighborSet[core]
        cluster = set(np.append(core_neighbors, [core]))
        cluster = cluster & unvisited
        unvisited = unvisited - cluster
        cluster_linda0 = []
        for i in list(cluster):
            cluster_linda0.append(eigvalues[i][0])  # 获取每个关键点协方差矩阵的最小特征值
        cluster_linda0 = np.asarray(cluster_linda0)
        NMS_OUTPUT = np.argmax(cluster_linda0)
        key_indices = np.append(key_indices, list(cluster)[NMS_OUTPUT])
    key_indices=key_indices[1:]

    return key_indices, neighborSet, eigvectors


tic = time.time()

query_pcd = poisson_sample(r'C:\Users\Admin\OneDrive\桌面\flange-coupling-2_step_1.stl')
mr = meshResolution(query_pcd)
target_pcd = poisson_sample(r'C:\Users\Admin\OneDrive\桌面\flange-coupling-2_step_1.stl', pymeshlab.AbsoluteValue(mr*0.94))

print(query_pcd.shape, target_pcd.shape)

qry_key_idx, qry_ns, qry_eigvec = computeISS(query_pcd, radius=3*mr)
tgt_key_idx, tgt_ns, tgt_eigvec = computeISS(target_pcd, radius=3*mr)

print(qry_key_idx.shape, tgt_key_idx.shape)

qry_LMA = getLMA(query_pcd, qry_eigvec, radius=7*mr)
tgt_LMA = getLMA(target_pcd, tgt_eigvec, radius=7*mr)

qry_DLFS = getDLFS(query_pcd, qry_LMA, qry_key_idx)
tgt_DLFS = getDLFS(target_pcd, tgt_LMA, tgt_key_idx)

qry_DLFS = np.reshape(qry_DLFS, (qry_DLFS.shape[0], -1))
tgt_DLFS = np.reshape(tgt_DLFS, (tgt_DLFS.shape[0], -1))

print(qry_DLFS.shape, tgt_DLFS.shape)

distances = cdist(qry_DLFS, tgt_DLFS, metric='canberra')
print(distances)
np.save('distances1.npy', distances)

print(time.time()-tic)
