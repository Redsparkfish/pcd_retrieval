import os
import time
from DLFS_calculation import computeDLFS
from bag_of_feature import *
from global_descriptor import *
import json
import stat


file = open('configuration.json', 'r', encoding='utf-8')
config = json.load(file)
data_dir = config['data_dir']
mode = config['mode']

key_indices_list, new_names, new_categories = computeDLFS(data_dir, mode)

size = new_names.shape[0]
batch_size = 500
num_clusters_batch = 50
current_batch = 0
meta_path = os.path.join(data_dir, 'meta.json')
meta = None
last_id = 0
if mode.lower() == 'stp':
    with open(meta_path, 'r+') as file:
        meta = json.load(file)
        file.close()
    bug_file = open(os.path.join(data_dir, 'bug_log.txt'), 'a')
    for d in meta:
        category = d['partType']
        name = d['partName']
        try:
            scale_par = param_desc(data_dir, category, name)
        except:
            bug_file.write(f'scale param for {name} failed.\n')
            scale_par = np.zeros(17, dtype=float)
        d['param_desc'] = scale_par
    meta = json.dumps(meta, indent=2, ensure_ascii=False)
    with open(os.path.join(data_dir, 'meta.json'), 'w') as outfile:
        outfile.write(meta)
        outfile.close()
    bug_file.close()
    exit()

if mode.lower() == 'meta':
    kmeans_list = np.load(os.path.join(data_dir, 'kmeans_list.npy'), allow_pickle=True).tolist()
    categories_list = np.load(os.path.join(data_dir, 'categories_list.npy'), allow_pickle=True).tolist()
    names_list = np.load(os.path.join(data_dir, 'names_list.npy'), allow_pickle=True).tolist()
    high_kmeans = np.load(os.path.join(data_dir, 'high_kmeans.npy'), allow_pickle=True)[0]
    meta = computeGlobalDescriptors(data_dir, high_kmeans, kmeans_list, categories_list, names_list, new_categories,
                                        new_names, last_id)
    meta = json.dumps(meta, indent=2, ensure_ascii=False)
    with open(os.path.join(data_dir, 'meta.json'), 'w') as outfile:
        outfile.write(meta)
        outfile.close()
    print('Descriptor calculation finished')

if os.path.exists(meta_path) and mode.lower() == 'update':
    with open(meta_path, 'r+') as file:
        meta = json.load(file)
        file.close()
    last_id = meta[-1].get('id') + 1
    kmeans_list = np.load(os.path.join(data_dir, 'kmeans_list.npy'), allow_pickle=True).tolist()
    categories_list = np.load(os.path.join(data_dir, 'categories_list.npy'), allow_pickle=True).tolist()
    names_list = np.load(os.path.join(data_dir, 'names_list.npy'), allow_pickle=True).tolist()
    last_num = len(names_list[-1])
    if last_num < batch_size:
        train_names = np.append(names_list[-1], new_names)
        train_categories = np.append(categories_list[-1], new_categories)
        size = train_names.shape[0]
        kmeans_list = kmeans_list[:-1]
        categories_list = categories_list[:-1]
        names_list = names_list[:-1]
else:
    train_names = new_names.copy()
    train_categories = new_categories.copy()
    kmeans_list = []
    categories_list = []
    names_list = []
init_list_len = len(kmeans_list)

tic0 = time.time()

while size >= batch_size:
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
if os.path.exists(meta_path):
    os.chmod(os.path.join(data_dir, 'names_list.npy'), stat.S_IWRITE)
    os.chmod(os.path.join(data_dir, 'categories_list.npy'), stat.S_IWRITE)
    os.chmod(meta_path, stat.S_IWRITE)

np.save(os.path.join(data_dir, 'kmeans_list.npy'), kmeans_list)
np.save(os.path.join(data_dir, 'names_list.npy'), names_list)
np.save(os.path.join(data_dir, 'categories_list.npy'), categories_list)

if len(kmeans_list) < 2:
    high_kmeans = kmeans_list[0]
else:
    high_kmeans = construct_high_codebook(kmeans_list)

if os.path.exists(meta_path):
    os.chmod(os.path.join(data_dir, 'high_kmeans.npy'), stat.S_IWRITE)
np.save(os.path.join(data_dir, 'high_kmeans.npy'), [high_kmeans])
print('Training finished in', (time.time() - tic0)/60, 'min. Initializing descriptor calculation.')

# kmeans_list = np.load(os.path.join(data_dir, 'kmeans_list.npy'), allow_pickle=True).tolist()
# categories_list = np.load(os.path.join(data_dir, 'categories_list.npy'), allow_pickle=True).tolist()
# names_list = np.load(os.path.join(data_dir, 'names_list.npy'), allow_pickle=True).tolist()
# high_kmeans = np.load(os.path.join(data_dir, 'high_kmeans.npy'), allow_pickle=True)[0]

categories_list = categories_list[init_list_len:]
names_list = names_list[init_list_len:]

tic1 = time.time()
meta = update_meta_bof(meta, data_dir, high_kmeans, kmeans_list)
new_meta = computeGlobalDescriptors(data_dir, high_kmeans, kmeans_list, categories_list, names_list, new_categories, new_names, last_id)
if meta:
    meta += new_meta
    os.chmod(os.path.join(data_dir, 'meta.json'), stat.S_IWRITE)
else:
    meta = new_meta
meta = json.dumps(meta, indent=2, ensure_ascii=False)
with open(os.path.join(data_dir, 'meta.json'), 'w') as outfile:
    outfile.write(meta)
    outfile.close()
print('Descriptor calculation finished in', (time.time() - tic1)/60, 'min.\n Total time consumed:', (time.time() - tic0)/60, 'min.')
