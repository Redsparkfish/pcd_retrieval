import scipy.spatial.distance
from scipy.spatial.distance import cdist
from DLFS_calculation import *
from global_descriptor import D2, sparseCoding
import json
import sys
import os


def isInMeta(mesh_path: str, meta):
    name = mesh_path.split('\\')[-1].split('.')[0]
    for d in meta:
        if d['partName'] == name:
            return d
    return False


def calcDescriptors(mesh, high_kmeans, kmeans_list):
    points = trimesh.sample.sample_surface(mesh, 30000)[0]
    mr = meshResolution(points) / 1.25
    key_indices, neighbor_indices, eigvectors = computeISS(points, rate=1, radius=2.5 * mr)
    LMA = getLMA(points, eigvectors, radius=7 * mr)
    DLFSs = getDLFS(points, LMA, key_indices, R=20 * mr)
    DLFSs = DLFSs.reshape(DLFSs.shape[0], DLFSs.shape[1]*DLFSs.shape[2])
    distribution = D2(points)
    bof_descriptor = sparseCoding(DLFSs, high_kmeans, kmeans_list)
    return distribution, bof_descriptor


def calcSimilarity(desc1: np.ndarray, desc2: np.ndarray):
    canberra_dist = scipy.spatial.distance.canberra(desc1, desc2)
    return np.square(np.exp(-(canberra_dist*0.020)**4))


def retrieval(query_desc, batch_descs, k=10):
    dists = cdist(query_desc.reshape(1, query_desc.shape[0]), batch_descs, metric='canberra')
    close_idx = np.argsort(dists[0])[:k + 1]
    return close_idx


def retrieve_test(meta, mesh_path, high_kmeans, kmeans_list, k=10):
    d = isInMeta(mesh_path, meta)
    if d:
        d2 = d['d2_desc']
        bof_desc = d['bof_desc']
        scale_par = d['param_desc']
    else:
        mesh = trimesh.load_mesh(mesh_path)
        d2, bof_desc = calcDescriptors(mesh, high_kmeans, kmeans_list)
        scale_par = np.zeros(17, dtype=float)

    query_desc = np.hstack((scale_par, d2, bof_desc))

    par_descs = np.vstack([meta[i]['param_desc'] for i in range(len(meta))])
    d2_list = np.vstack([meta[i]['d2_desc'] for i in range(len(meta))])
    bof_descs = np.vstack([meta[i]['bof_desc'] for i in range(len(meta))])
    fuse_descs = np.hstack((par_descs, d2_list, bof_descs))

    dists = cdist(query_desc.reshape(1, query_desc.shape[0]), fuse_descs, metric='canberra')
    close_idx = np.argsort(dists[0])[:k + 1]

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
    mesh_path = sys.argv[1]
    partType = sys.argv[2].lower()
    clientInfo = sys.argv[3].lower()
    k = int(sys.argv[4])

    configPath = sys.argv[5]

    file = open(configPath, 'r')
    config = json.load(file)
    data_dir = config['data_dir']

    with open(os.path.join(data_dir, 'meta.json'), 'r') as metafile:
        meta = json.load(metafile)

    # meta = np.load(os.path.join(data_dir, 'meta.npy'), allow_pickle=True)
    if partType.lower() != 'all':
        meta = [d for d in meta if d['partType'] == partType]
    high_kmeans = np.load(os.path.join(data_dir, 'high_kmeans.npy'), allow_pickle=True)[0]
    kmeans_list = np.load(os.path.join(data_dir, 'kmeans_list.npy'), allow_pickle=True)
    results = retrieve_test(meta, mesh_path, high_kmeans, kmeans_list, k)
    results = json.dumps({"stlList": results}, ensure_ascii=False)
    results = eval(repr(results).replace('\\\\', '/'))
    results = eval(repr(results).replace('//', '/'))
    print(results)
