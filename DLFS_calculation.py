import os
import time
# import pymeshlab
import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors


# def simplification(path):
#     ms = pymeshlab.MeshSet()
#     ms.load_new_mesh(path)
#     ms.meshing_decimation_clustering()
#     ms.save_current_mesh(path)


def getKNeighbors(points, k):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(points)
    k_neighbor_indices = knn.kneighbors(return_distance=False)
    return k_neighbor_indices


def getNeighbors(points, radius):
    nn = NearestNeighbors(radius=radius)
    nn.fit(points)
    neighbor_indices = nn.radius_neighbors(return_distance=False)
    return neighbor_indices


def computeISS(points, t1=0.95, t2=0.95, rate=2, radius=7 / 3):
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
    threshold = np.median(eigvalues, axis=0)[0] * rate
    sorted_idx = np.argsort(eigvalues[:, 0])
    choice = np.random.randint(0, point_size, 300)
    key_indices = np.append(sorted_idx[-700:], sorted_idx[choice])
    # key_indices = sorted_idx[-1000:]
    '''
    def acceptIndex(index):
        i = index[0]
        print(len(neighborSet[i]))
        if t1 * eigvalues[i][1] > eigvalues[i][0] and t2 * eigvalues[i][2] > eigvalues[i][1]:
            if eigvalues[i][0] > threshold:
                return True
        return False
    indices = index_set[np.apply_along_axis(acceptIndex, -1, index_set.reshape(point_size, 1))]
    unvisited = set(indices)
    key_indices = np.zeros(1)

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
    '''
    return key_indices, neighborSet, eigvectors


def getLMA(points, eigvectors, radius=7):
    size = points.shape[0]
    neighbor_indices = getNeighbors(points, radius)

    def calcLRA(index):
        i = index[0]
        LRA = eigvectors[i]
        d_vectors = points[neighbor_indices[i]] - points[i]
        sum_d_vector = np.sum(d_vectors, axis=0)
        if np.dot(eigvectors[i], sum_d_vector) < 0:
            LRA = -LRA
        LRA /= np.linalg.norm(LRA)
        return LRA

    LRAs = np.apply_along_axis(calcLRA, -1, np.arange(size).reshape(size, 1))
    return LRAs


def getDLFS(points, LMA, key_indices, N=[5, 9, 12, 12, 15], lamda=[0.6, 1.1, 1, 0.9], R=20):
    neighbor_indices = getNeighbors(points, radius=R)

    def calcDLFS(key_index):
        key_index = key_index[0]
        LRA = LMA[key_index]
        pq = points[neighbor_indices[key_index]] - points[key_index]
        distances = np.linalg.norm(np.cross(LRA, pq), axis=1)
        lh = np.sum((LRA * pq), axis=1) + R
        cross = np.cross(LRA, pq)
        cross_norms = np.linalg.norm(np.cross(LRA, pq), axis=1)
        cross_nonzero = cross_norms > 0
        cross_norms.reshape(cross.shape[0], 1)
        cross_temp = np.copy(cross_norms[cross_nonzero])
        cross[cross_nonzero] = cross[cross_nonzero] / cross_temp.reshape(cross_temp.shape[0], 1)

        temp = np.sum(cross * LMA[neighbor_indices[key_index]], axis=1)
        temp[temp > 1] = 1
        temp[temp < -1] = -1
        alpha = np.arccos(temp)

        LMA1 = LMA[neighbor_indices[key_index]] - temp.reshape([temp.shape[0], 1]) * np.cross(LRA, pq)
        LMA1_norm = np.linalg.norm(LMA1, axis=1)
        LMA1_nonzero = LMA1_norm > 0
        LMA1[LMA1_nonzero] = LMA1[LMA1_nonzero] / LMA1_norm[LMA1_nonzero].reshape(LMA1_norm[LMA1_nonzero].shape[0], 1)

        temp2 = np.sum(LRA * LMA1, axis=1)
        temp2[temp2 > 1] = 1
        temp2[temp2 < -1] = -1
        beta = np.arccos(temp2)

        temp3 = np.sum(pq * LMA1, axis=1) / np.linalg.norm(pq, axis=1)
        temp3[temp3 > 1] = 1
        temp3[temp3 < -1] = -1
        gamma = np.arccos(np.sum(pq * LMA1, axis=1) / np.linalg.norm(pq, axis=1))

        H_lh = np.empty((N[0], N[1]))
        H_alpha = np.empty((N[0], N[2]))
        H_beta = np.empty((N[0], N[3]))
        H_gamma = np.empty((N[0], N[4]))
        for i in range(N[0]):
            indices = np.logical_and(i * R / N[0] < distances, distances <= (i + 1) * R / N[0])
            H_lh[i] = np.histogram(lh[indices], bins=N[1], range=(0, 2 * R))[0]
            if np.sum(H_lh[i]) != 0:
                H_lh[i] /= np.sum(H_lh[i])

            H_alpha[i] = np.histogram(alpha[indices], bins=N[2], range=(0, np.pi))[0]
            if np.sum(H_alpha[i]) != 0:
                H_alpha[i] /= np.sum(H_alpha[i])

            H_beta[i] = np.histogram(beta[indices], bins=N[3], range=(0, np.pi))[0]
            if np.sum(H_beta[i]) != 0:
                H_beta[i] /= np.sum(H_beta[i])

            H_gamma[i] = np.histogram(gamma[indices], bins=N[4], range=(0, np.pi))[0]
            if np.sum(H_gamma[i]) != 0:
                H_gamma[i] /= np.sum(H_gamma[i])

        result = np.hstack((lamda[0] * H_lh,
                            lamda[1] * H_alpha,
                            lamda[2] * H_beta,
                            lamda[3] * H_gamma))
        return result

    histograms = np.apply_along_axis(calcDLFS, -1, key_indices.reshape(key_indices.shape[0], 1))

    return histograms


def meshResolution(points):
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(points)
    distances, indices = nn.kneighbors(points)
    mr = np.sum(distances) / distances.shape[0]
    return mr


def computeDLFS(data_dir, mode='update'):
    new_names = []
    new_categories = []
    key_indices_list = []
    tic = time.time()
    for folder in os.listdir(data_dir):
        # Skip any files in the data directory
        if not os.path.isdir(os.path.join(data_dir, folder)):
            continue
        # Skip the update directory
        if folder.lower() in ['update', 'temp_update']:
            continue
        category_tic = time.time()
        for filename in os.listdir(os.path.join(data_dir, folder, 'STL')):
            if not filename.endswith('.stl'):
                continue
            if os.path.exists(os.path.join(data_dir, folder, 'DCT', filename[:-4]+'.npy')):
                if mode.lower() != 'update':
                    new_names.append(filename[:-4])
                    new_categories.append(folder)
                continue
            new_names.append(filename[:-4])
            new_categories.append(folder)
            path = os.path.join(data_dir, folder, 'STL', filename)
            mesh = trimesh.load_mesh(path)
            points = trimesh.sample.sample_surface(mesh, 50000)[0]
            points = np.unique(points, axis=0)
            if not os.path.exists(os.path.join(data_dir, folder, 'PCD')):
                os.makedirs(os.path.join(data_dir, folder, 'PCD'))
            np.save(os.path.join(data_dir, folder, 'PCD', filename[:-4]+'.npy'), points)
            mr = meshResolution(points)

            # Extract ISS keypoints
            print(filename)
            key_indices, neighbor_indices, eigvectors = computeISS(points, rate=1, radius=2.5 * mr)

            # Extract DLFS features from the mesh
            LMA = getLMA(points, eigvectors, radius=7 * mr)
            DLFSs = getDLFS(points, LMA, key_indices, R=20 * mr)
            DLFSs = DLFSs.reshape(DLFSs.shape[0], DLFSs.shape[1]*DLFSs.shape[2])
            # Append the features to the list

            if not os.path.exists(os.path.join(data_dir, folder, 'DCT')):
                os.makedirs(os.path.join(data_dir, folder, 'DCT'))
            np.save(os.path.join(data_dir, folder, 'DCT', filename[:-4] + '.npy'), DLFSs)
            key_indices_list.append(key_indices)

    print('Calculation finished in', (time.time() - tic) / 60, 'min.')
    return np.array(key_indices_list), np.array(new_names), np.array(new_categories)


# computeDLFS(r'C:\Users\Admin\模型分类_mesh')

# mesh = trimesh.load_mesh(r'C:/Users/Admin/模型分类/新建文件夹/8 (2).stl')
# points = trimesh.sample.sample_surface(mesh, 10000)[0]
# mr = meshResolution(points)
# key_indices, neighborSet, eigvectors = computeISS(points, radius=7/3*mr)
# keypoints = points[key_indices]
# pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(keypoints))
# open3d.visualization.draw([pcd])
