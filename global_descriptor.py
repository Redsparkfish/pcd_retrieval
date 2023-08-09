from scipy.spatial.distance import cdist
import numpy as np
import os
from stp_par import *


def D2(points: np.ndarray, b_num=100):
    size = points.shape[0]
    permutation = np.random.permutation(size)
    dvector = points - points[permutation]
    distances = np.linalg.norm(dvector, axis=1)
    return np.histogram(distances, bins=b_num)[0] / size


def sparseCoding(DLFS: np.ndarray, high_kmeans, kmeans_list):
    all_centers = np.vstack([kmeans_list[i].cluster_centers_ for i in range(len(kmeans_list))])
    distances = cdist(DLFS, all_centers)
    labels = np.argmin(distances, axis=1)
    high_labels = high_kmeans.labels_

    num_clusters = high_kmeans.cluster_centers_.shape[0]
    sparse_descriptor = np.zeros(num_clusters)
    for label in labels:
        sparse_descriptor[high_labels[label]] += 1

    return sparse_descriptor / np.sum(sparse_descriptor)


def computeStrength(x, GMM_mean: np.ndarray, GMM_cov: np.ndarray, GMM_weights):
    K = GMM_mean.shape[0]
    strength = np.zeros((K, x.shape[0]))
    A = np.zeros(x.shape[0])
    for j in range(K):
        strength[j] = GMM_weights[j] * np.exp(-np.diag(np.linalg.multi_dot((x - GMM_mean[j], np.diag(1 / GMM_cov[j]), (x - GMM_mean[j]).T)) / 2))
    A = np.sum(strength, axis=0)
    for i in range(x.shape[0]):
        strength[:, i] = strength[:, i] / A[i]
    return strength  # strength.shape = (K, N)


def fisherVector(DLFS: np.ndarray, GMM_mean: np.ndarray, GMM_cov: np.ndarray, GMM_weights: np.ndarray, strength: np.ndarray):
    K = GMM_mean.shape[0]
    d = GMM_mean.shape[1]
    N = DLFS.shape[0]
    u = np.empty((K, d))
    v = np.empty((K, d))

    for k in range(K):
        u[k] = np.diag(1 / np.sqrt(GMM_cov[k])) @ np.sum(strength[k].reshape(-1, 1) * (DLFS-GMM_mean[k]), axis=0) / (N*np.sqrt(GMM_weights[k]))
        v[k] = np.diag(1 / GMM_cov[k]) @ np.sum(strength[k].reshape(-1, 1) * (np.square(DLFS-GMM_mean[k])-1), axis=0) / (N*np.sqrt(2*GMM_weights[k]))

    return u, v


def param_desc(data_dir, category, name):
    stp_path = ''
    if os.path.exists(os.path.join(data_dir, category, 'STP', name + '.stp')):
        stp_path = os.path.join(data_dir, category, 'STP', name + '.stp')
    elif os.path.exists(os.path.join(data_dir, category, 'STEP', name + '.stp')):
        stp_path = os.path.join(data_dir, category, 'STEP', name + '.stp')
    elif os.path.exists(os.path.join(data_dir, category, 'STEP', name + '.step')):
        stp_path = os.path.join(data_dir, category, 'STEP', name + '.step')
    elif os.path.exists(os.path.join(data_dir, category, 'STP', name + '.step')):
        stp_path = os.path.join(data_dir, category, 'STP', name + '.step')
    else:
        scale_par = np.zeros(17, dtype=float)
    if os.path.exists(stp_path):
        try:
            par, scale_par = get_par(read_step_file(stp_path))
        except:
            raise Exception("STP parameters calculation failed, using zeros instead...")

    return scale_par


def computeGlobalDescriptors(data_dir, high_kmeans, kmeans_list, categories_list, names_list, new_categories, new_names, last_id=0):
    meta = []
    k = last_id
    bug_file = open(os.path.join(data_dir, 'bug_log.txt'), 'a')
    for i in range(len(names_list)):
        for j in range(len(names_list[i])):
            name = names_list[i][j]
            category = categories_list[i][j]
            if name not in new_names:
                continue
            elif category not in new_categories[new_names == name]:
                continue
            pcd_path = os.path.join(data_dir, category, 'PCD', name + '.npy')
            points = np.load(pcd_path)
            d2_desc = D2(points)
            DLFS = np.load(os.path.join(data_dir, category, 'DCT', name + '.npy'))
            bof_desc = sparseCoding(DLFS, high_kmeans, kmeans_list)
            try:
                scale_par = param_desc(data_dir, category, name)
            except:
                bug_file.write(f'scale param for {name} failed.\n')
                scale_par = np.zeros(17, dtype=float)

            info_dist = {'id': k, 'partType': category, 'partName': name, 'd2_desc': d2_desc.tolist(),
                         'bof_desc': bof_desc.tolist(), 'param_desc': scale_par.tolist()}
            meta.append(info_dist)
            print(k, name)
            k += 1
    bug_file.close()
    return meta


def update_meta_bof(meta, data_dir, high_kmeans, kmeans_list):
    for d in meta:
        DLFS = np.load(os.path.join(data_dir, d['partType'], 'DCT', d['partName'] + '.npy'))
        bof_desc = sparseCoding(DLFS, high_kmeans, kmeans_list).tolist()
        d['bof_desc'] = bof_desc
        print(f"bof_desc of {d['partName']} updated")
    return meta
