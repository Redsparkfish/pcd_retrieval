import pymysql
import sys
import trimesh
from scipy.spatial.distance import cdist
from DLFS_calculation import *
from global_descriptor import D2, sparseCoding
import json
import os
import pandas as pd


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


def retrieve_test(mesh_path, k=10):
    mesh = trimesh.load_mesh(mesh_path)
    d2, bof_desc = calcDescriptors(mesh, codebook)
    descriptor = np.hstack((d2, bof_desc))
    fuse_descs = np.hstack((d2_list, descriptors))

    dists = cdist(descriptor.reshape(1, descriptor.shape[0]), fuse_descs, metric='canberra')

    close_idx = np.argsort(dists[0])[:k + 1]
    return close_idx


def pr_curve(retrieval_results, category_name, category_total):
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


mesh_path = sys.argv[1]
category = sys.argv[2].lower()
if category != 'all':
    category = category[0].upper() + category[1:].lower()
clientInfo = sys.argv[3].lower()
k = int(sys.argv[4])

configPath = sys.argv[5]

file = open(configPath, 'r')
config = json.load(file)
data_dir = config['data_dir']
codebook = np.load(data_dir+'/codebook.npy')


db = pymysql.connect(host=config['db_host'], user=config['db_user'], password=config['db_password'], database=config['db_database'], cursorclass=pymysql.cursors.DictCursor)
cursor = db.cursor()
try:
    with db.cursor() as cursor:
        # Read some records
        if category != 'all' and clientInfo != 'all':
            sql = "SELECT `partType`, `partName`, `distributeDesc`, `bofDesc`, `clientInfo` FROM `part_desc` " \
                  "WHERE `partType` = %s AND `clientInfo` = %s"
            cursor.execute(sql, (category, clientInfo))
        elif category != 'all' and clientInfo == 'all':
            sql = "SELECT `partType`, `partName`, `distributeDesc`, `bofDesc` FROM `part_desc` " \
                  "WHERE `partType` = %s"
            cursor.execute(sql, category)
        elif category == 'all' and clientInfo != 'all':
            sql = "SELECT `partType`, `partName`, `distributeDesc`, `bofDesc`, `clientInfo` FROM `part_desc` " \
                  "WHERE `clientInfo` = %s"
            cursor.execute(sql, clientInfo)
        elif category == 'all' and clientInfo == 'all':
            sql = "SELECT `partType`, `partName`, `distributeDesc`, `bofDesc` FROM `part_desc`"
            cursor.execute(sql)

        result = cursor.fetchall()

finally:
    db.close()

df = pd.DataFrame(result)
categories = df['partType'].tolist()
names = df['partName'].to_numpy()
d2_list = np.array(df['distributeDesc'].apply(json.loads).to_list())
descriptors = np.array(df['bofDesc'].apply(json.loads).to_list())
idx = retrieve_test(mesh_path, k)
results = '{\n' + json.dumps('stlList') + ':[\n'
for i in range(k):
    results += json.dumps({"index": i, "name": names[idx[i]], "path": (os.path.join(data_dir, categories[idx[i]], 'STL', names[idx[i]]))})
    if i < k-1:
        results += ','
    results += '\n'
results += ']\n}'

results = eval(repr(results).replace('\\\\', '/'))
results = eval(repr(results).replace('//', '/'))
print(results)
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


# def retrieve(query_name, k):
#     d2_list = np.load('C:/Users/Admin/CAD_parts/d2_list.npy')
#     size = d2_list.shape[0]
#     n = 600
#     nn = NearestNeighbors(n_neighbors=n+1, metric='canberra')
#     nn.fit(d2_list)
#     distances, indices_list = nn.kneighbors(d2_list)
#
#     names = np.load('C:/Users/Admin/CAD_parts/names.npy')
#
#     index = (names == query_name)
#     distances = cdist(descriptor_set[index], descriptor_set[indices_list[index]][0], metric='canberra')
#     idx = np.argsort(distances[0])[:k+1]
#     closest = names[indices_list[index][0][idx]]
#
#
#     # summary = 0
#     #
#     # for i in range(size):
#     #     str_list = names[i].split('_')
#     #     n = len(str_list)
#     #     category = ''
#     #     for j in range(n-1):
#     #         category += str_list[j] + '_'
#     #     name = names[i]
#     #     # print(category)
#     #     # print(name)
#     #     distances = cdist(descriptor_set[i].reshape(1, descriptor_set[i].shape[0]), descriptor_set[indices_list[i]], metric='canberra')
#     #     idx = np.argsort(distances[0])[:k+1]
#     #     closest = names[indices_list[i][idx[:k + 1]]]
#     #     if category == '':
#     #         print(name)
#     #         print(closest)
#     #     print(name, ':')
#     #     for j in range(k+1):
#     #         if closest[j].startswith(category) and closest[j] != name:
#     #             print(closest[j])
#     #             summary += 1
#     #         #elif closest[j] != name:
#     #         #    print(name,  closest[j], '\n')
#     #     print('')
#     #
#     # print(summary/(k * size))
#
#     return closest

# nn = NearestNeighbors(n_neighbors=k+1, metric='canberra')
# nn.fit(descriptors)
# distances, indices_list = nn.kneighbors(descriptors)
#
# summary = 0
# for i in range(size):
#     str_list = names[i].split('_')
#     n = len(str_list)
#     category = ''
#     for j in range(n - 1):
#         category += str_list[j] + '_'
#     name = names[i]
#     closest = names[indices_list[i]][1:]
#     for j in range(k):
#         if closest[j].startswith(category):
#             summary += 1
# print(summary/(k * size))
# '''
# '''
# input_name = 'Bearings_24161c0b-afc7-449c-a803-62ae4e88cc73.stl'
# ft = 58
# result = retrieve(input_name, 2*ft+1)
# summary = 0
# category = 'Bearings'
# for j in range(2*ft+1):
#     if result[j].startswith(category):
#         summary += 1
#
# print(summary/ft)
