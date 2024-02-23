import numpy as np
import scipy.spatial.distance
from scipy.spatial.distance import cdist
from DLFS_calculation import *
from global_descriptor import D2, sparseCoding
import json
import sys
import argparse
import os
import time


def isInMeta(mesh_path: str, meta):
    name = mesh_path.split('\\')[-1].split('.')[0]
    for d in meta:
        if d['partName'] == name:
            return d
    return False


def calcDescriptors(mesh, high_kmeans, kmeans_list):

    points = trimesh.sample.sample_surface(mesh, 30000)[0]
    mr = meshResolution(points) / 1.25
    tic00 = time.time()
    key_indices, neighbor_indices, eigvectors = computeISS(points, rate=1, radius=2.5 * mr)
    tic01 = time.time()
    LMA = getLMA(points, eigvectors, radius=7 * mr)
    DLFSs = getDLFS(points, LMA, key_indices, R=20 * mr)
    tic02 = time.time()
    DLFSs = DLFSs.reshape(DLFSs.shape[0], DLFSs.shape[1]*DLFSs.shape[2])
    distribution = D2(points)
    bof_descriptor = sparseCoding(DLFSs, high_kmeans, kmeans_list)
    tic03 = time.time()

    print('ISS time:', tic01-tic00)
    print('DLFS time:', tic02-tic01)
    print('BOF time:', tic03-tic02)
    return distribution, bof_descriptor


def calcSimilarity(desc1: np.ndarray, desc2: np.ndarray):
    dist = custom_dist(desc1, desc2)
    sim = np.exp(-(0.2*dist**4))
    if sim > 1:
        sim = 0
    return sim


def custom_dist(p1, p2):
    p1_scale_par = p1[:17]
    p1_bof_desc = p1[17:-100]
    p1_d2 = p1[-100:]

    p2_scale_par = p2[:17]
    p2_bof_desc = p2[17:-100]
    p2_d2 = p2[-100:]

    distance = 0
    distance += 0.1*scipy.spatial.distance.canberra(p1_scale_par, p2_scale_par)
    distance += scipy.spatial.distance.cityblock(p1_bof_desc, p2_bof_desc)
    distance += scipy.spatial.distance.cityblock(p1_d2, p2_d2)
    return distance


def retrieve_test(meta, mesh_path, high_kmeans, kmeans_list, k=10):
    d = isInMeta(mesh_path, meta)
    if d:
        d2 = d['d2_desc']
        bof_desc = d['bof_desc']
        scale_par = np.array(d['param_desc'])
    else:
        mesh = trimesh.load_mesh(mesh_path)
        d2, bof_desc = calcDescriptors(mesh, high_kmeans, kmeans_list)
        scale_par = np.zeros(17, dtype=float)

    query_desc = np.hstack((scale_par, d2, bof_desc))
    stl_desc = np.hstack((d2, bof_desc))

    stl_dists = cdist(stl_desc.reshape(1, stl_desc.shape[0]), stl_descs, metric="cityblock")
    par_dists = 0.1*cdist(scale_par.reshape(1, scale_par.shape[0]), par_descs, metric='canberra')
    dists = cdist(query_desc.reshape(1, query_desc.shape[0]), fuse_descs, metric=custom_dist)
    s = np.vstack((stl_dists[0], par_dists[0], dists[0])).T
    print(np.sum(s, axis=0))
    close_idx = np.argsort(dists[0])

    similarities = [calcSimilarity(query_desc, desc) for desc in fuse_descs]

    results = [{"index": meta[close_idx[i]]['id'], "similarity": str(similarities[close_idx[i]])[:6], "clientInfo": "",
                "partType": meta[close_idx[i]]['partType'], "partName": meta[close_idx[i]]['partName'],
                "path": os.path.join(data_dir, meta[close_idx[i]]['partType'], 'STL', meta[close_idx[i]]['partName']+'.stl')}
               for i in range(k)]

    return results


def pr_curve(retrieval_results, category_name, category_total):
    j = 0
    precision = []
    recall = []
    n = len(retrieval_results)
    for i, result in enumerate(retrieval_results):
        if result.startswith(category_name):
            j += 1
        precision.append(j / (i+1))
        recall.append(j / category_total)
    return np.array(precision), np.array(recall)


if __name__ == '__main__':
    tic = time.time()

    parser = argparse.ArgumentParser(description="Retrieval Arguments")
    parser.add_argument('--meshPath', '-mp', type=str,
                        default=r"C:\Users\Admin\Bearings_00ed2536-3d80-4f07-8851-4f49f1606498",
                        help="the path of the stl file")
    parser.add_argument('--partType', '-ptt', type=str, default='all', help="part type attribute")
    parser.add_argument('--clientInfo', '-ci', type=str, default='all', help="client info attribute")
    parser.add_argument('--retrievalNum', '-k', type=int, default=10, help='desired number of retrieval results')
    parser.add_argument('--configPath', '-cp', type=str, default='configuration.json', help='the path of the configuration file')

    args = parser.parse_args()
    mesh_path = args.meshPath
    partType = args.partType
    clientInfo = args.clientInfo
    k = args.retrievalNum
    configPath = args.configPath

    file = open(configPath, 'r', encoding='utf-8')
    config = json.load(file)
    data_dir = config['data_dir']

    with open(os.path.join(data_dir, 'meta.json'), 'r') as metafile:
        meta = json.load(metafile)
        metafile.close()

    # meta = np.load(os.path.join(data_dir, 'meta.npy'), allow_pickle=True)
    if partType.lower() != 'all':
        meta = [d for d in meta if d['partType'] == partType]
    high_kmeans = np.load(os.path.join(data_dir, 'high_kmeans.npy'), allow_pickle=True)[0]
    kmeans_list = np.load(os.path.join(data_dir, 'kmeans_list.npy'), allow_pickle=True)

    par_descs = np.vstack([meta[i]['param_desc'] for i in range(len(meta))])
    d2_list = np.vstack([meta[i]['d2_desc'] for i in range(len(meta))])
    bof_descs = np.vstack([meta[i]['bof_desc'] for i in range(len(meta))])
    stl_descs = np.hstack((d2_list, bof_descs))
    fuse_descs = np.hstack((par_descs, d2_list, bof_descs))

    tic1 = time.time()

    results = retrieve_test(meta, mesh_path, high_kmeans, kmeans_list, k)
    results = json.dumps({"stlList": results}, indent=2, ensure_ascii=False)
    results = eval(repr(results).replace('\\\\', '/'))
    results = eval(repr(results).replace('//', '/'))
    print(results)

    print("reading time:", tic1-tic)
    print("retrieving time:", time.time()-tic1)
