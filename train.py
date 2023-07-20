import time
from DLFS_calculation import computeDLFS
from bag_of_feature import *
from global_descriptor import *
import json


file = open('configuration.json', 'r', encoding='utf-8')
config = json.load(file)
data_dir = config['data_dir']
mode = config['mode']
update_dir = os.path.join(data_dir, 'update')

key_indices_list, train_names, train_categories = computeDLFS(data_dir, mode)

size = train_names.shape[0]
batch_size = 500
num_clusters_batch = 50
current_batch = 0
if os.path.exists(os.path.join(data_dir, 'kmeans_list.npy')) and mode.lower() == 'update':
    kmeans_list = np.load(os.path.join(data_dir, 'kmeans_list.npy'), allow_pickle=True).tolist()
    print(kmeans_list)
    categories_list = np.load(os.path.join(data_dir, 'categories_list.npy'), allow_pickle=True).tolist()
    names_list = np.load(os.path.join(data_dir, 'names_list.npy'), allow_pickle=True).tolist()
    last_num = len(kmeans_list[-1].cluster_centers_)
    if last_num < batch_size:
        train_names = np.append(names_list[-1], train_names)
        train_categories = np.append(categories_list[-1], train_categories)
        kmeans_list = kmeans_list[:-1]
        names_list = names_list[:-1]
        categories_list = categories_list[:-1]

else:
    kmeans_list = []
    categories_list = []
    names_list = []
    train_names = train_names
    train_categories = train_categories

batch_num = 0
tic0 = time.time()
while size >= batch_size:
    batch_num += 1
    DLFS_set = []
    category_set = []
    name_set = []
    for i in range(batch_size):
        idx = batch_size * current_batch + i
        DLFS_set.append(np.load(os.path.join(data_dir, train_categories[idx], 'DCT', train_names[idx] + '.npy')))
        category_set.append(train_categories[idx])
        name_set.append(train_names[idx])
    bof = np.concatenate(DLFS_set)
    print('Start Kmeans clustering.')
    kmeans = construct_codebook(bof, num_clusters=num_clusters_batch)
    kmeans_list.append(kmeans)
    categories_list.append(category_set)
    names_list.append(name_set)
    size -= batch_size
    current_batch += 1

if size != 0:
    batch_num += 1
    DLFS_set = []
    category_set = []
    name_set = []
    for i in range(size):
        idx = batch_size * current_batch + i
        DLFS_set.append(np.load(os.path.join(data_dir, train_categories[idx], 'DCT', train_names[idx]+'.npy')))
        category_set.append(train_categories[idx])
        name_set.append(train_names[idx])
    bof = np.concatenate(DLFS_set)
    print('Start Kmeans clustering.')
    kmeans = construct_codebook(bof, num_clusters=num_clusters_batch)
    kmeans_list.append(kmeans)
    categories_list.append(category_set)
    names_list.append(name_set)

names_list = np.array(names_list, dtype='object')
categories_list = np.array(categories_list, dtype='object')
np.save(os.path.join(data_dir, 'kmeans_list.npy'), kmeans_list)
np.save(os.path.join(data_dir, 'names_list.npy'), names_list)
np.save(os.path.join(data_dir, 'categories_list.npy'), categories_list)
# kmeans_list = np.load(os.path.join(data_dir, 'kmeans.npy'), allow_pickle=True)
# categories_list = np.load(os.path.join(data_dir, 'categories_list.npy'), allow_pickle=True)
# names_list = np.load(os.path.join(data_dir, 'names_list.npy'), allow_pickle=True)

if batch_num < 2:
    high_kmeans = kmeans_list[0]
else:
    high_kmeans = construct_high_codebook(kmeans_list)
np.save(os.path.join(data_dir, 'high_kmeans'), [high_kmeans])
print('Training finished in', (time.time() - tic0)/60, 'min. Initializing descriptor calculation.')

tic1 = time.time()
meta = json.dumps(computeGlobalDescriptors(data_dir, high_kmeans, kmeans_list, categories_list, names_list), indent=2, ensure_ascii=False)
with open(os.path.join(data_dir, 'meta.json'), 'w') as outfile:
    outfile.write(meta)
print('Descriptor calculation finished in', (time.time() - tic1)/60, 'min. Total time:', (time.time() - tic0)/60, 'min.')
