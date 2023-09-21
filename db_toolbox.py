import os.path
import time
import sys
from DLFS_calculation import computeDLFS
from bag_of_feature import *
from global_descriptor import *
import json
import shutil


def open_json(json_file, mode='r'):
    with open(json_file, mode=mode) as file:
        result = json.load(file)
        file.close()
    return result


def rm_dup(meta):
    partTypes = [d['partType'] for d in meta]
    partNames = [d['partName'] for d in meta]
    i = 0
    while i < len(meta):
        if partNames[i] in partNames[i+1:]:
            if partTypes[i+1:][partNames[i+1:].index(partNames[i])] == partTypes[i]:
                meta.pop(i)
                partNames.pop(i)
                partTypes.pop(i)
            else:
                i += 1
        else:
            i += 1
    i = 0
    for d in meta:
        d['id'] = i
        i += 1
    return meta


def get_kmeans_list(data_dir, train_partTypes, train_partNames, batch_size=500, num_clusters_batch=50):
    size = train_partNames.shape[0]
    current_batch = 0
    new_kmeans_list = []
    while size >= batch_size:
        DLFS_set = []
        for i in range(batch_size):
            idx = batch_size * current_batch + i
            DLFS_set.append(np.load(os.path.join(data_dir, train_partTypes[idx], 'DCT', train_partNames[idx] + '.npy')))
        bof = np.concatenate(DLFS_set)
        print('Start Kmeans clustering.')
        kmeans = construct_codebook(bof, num_clusters=num_clusters_batch)
        new_kmeans_list.append(kmeans)
        size -= batch_size
        current_batch += 1

    if size != 0:
        DLFS_set = []
        for i in range(size):
            idx = batch_size * current_batch + i
            DLFS_set.append(np.load(os.path.join(data_dir, train_partTypes[idx], 'DCT', train_partNames[idx] + '.npy')))
        bof = np.concatenate(DLFS_set)
        print('Start Kmeans clustering.')
        kmeans = construct_codebook(bof, num_clusters=num_clusters_batch)
        new_kmeans_list.append(kmeans)
    return new_kmeans_list


file = open('configuration.json', 'r', encoding='utf-8')
config = json.load(file)
data_dir = config['data_dir']
mode = config['mode']

feasible_modes = ['start', 'start stp', 'update', '!update', '!update stp', 'update stp', 'temp_update', 'temp_update stp', 'stp', 'delete']
if mode not in feasible_modes:
    raise Exception(f'mode value error. Possible modes: {feasible_modes}')
meta_path = os.path.join(data_dir, 'meta.json')
residuals_path = os.path.join(data_dir, 'residuals.json')
meta = None
last_id = 0
batch_size = 500
num_clusters_batch = 50
tic0 = time.time()


def temp_update():
    if not os.path.exists(meta_path):
        raise Exception("'meta.json does not exist.")
    meta = open_json(meta_path, mode='r')
    last_id = meta[-1].get('id') + 1
    kmeans_list = np.load(os.path.join(data_dir, 'kmeans_list.npy'), allow_pickle=True).tolist()
    high_kmeans = np.load(os.path.join(data_dir, 'high_kmeans.npy'), allow_pickle=True)[0]

    update_dir = os.path.join(data_dir, 'temp_update')
    key_indices_list, new_partNames, new_partTypes = computeDLFS(update_dir, mode)
    new_meta = computeGlobalDescriptors(update_dir, high_kmeans, kmeans_list, new_partTypes, new_partNames, last_id)
    meta += new_meta
    meta = rm_dup(meta)
    meta = json.dumps(meta, indent=2, ensure_ascii=False)
    with open(os.path.join(data_dir, 'meta.json'), 'w') as outfile:
        outfile.write(meta)
        outfile.close()

    for partType in np.unique(new_partTypes):
        src = os.path.join(update_dir, partType)
        dest1 = os.path.join(data_dir, 'update', partType)
        dest2 = os.path.join(data_dir, partType)
        shutil.copytree(src, dest1, dirs_exist_ok=True)
        shutil.copytree(src, dest2, dirs_exist_ok=True)
        shutil.rmtree(src)
        os.mkdir(src)
        os.mkdir(src + '/STL')
        os.mkdir(src + '/STP')


def stp():
    meta = open_json(meta_path)
    bug_file = open(os.path.join(data_dir, 'bug_log.txt'), 'a')
    tic = time.time()
    for d in meta:
        category = d['partType']
        name = d['partName']
        if d.get('param_desc') and d['param_desc'] != np.zeros(17).tolist():
            print(name, 'param_desc exists.')
            continue
        try:
            scale_par = param_desc(data_dir, category, name)
        except:
            bug_file.write(f'scale param for {name} failed.\n')
            scale_par = np.zeros(17, dtype=float).tolist()
        print(name, 'param_desc calculated.')
        d['param_desc'] = scale_par.tolist()
    meta = json.dumps(meta, indent=2, ensure_ascii=False)
    with open(os.path.join(data_dir, 'meta.json'), 'w') as outfile:
        outfile.write(meta)
        outfile.close()
    bug_file.close()
    print('param_descs calculation finished in', (time.time()-tic)/60, 'min.')


def update():
    if not os.path.exists(meta_path):
        raise Exception("'meta.json does not exist.")
    meta = open_json(meta_path, 'r')
    kmeans_list = np.load(os.path.join(data_dir, 'kmeans_list.npy'), allow_pickle=True).tolist()
    if os.path.exists(residuals_path):
        residuals = open_json(residuals_path)
    else:
        partTypes = [d['partType'] for d in meta]
        partNames = [d['partName'] for d in meta]
        while len(partTypes) >= batch_size:
            partTypes = partTypes[batch_size:]
            partNames = partNames[batch_size:]
        residuals = []
        for i in range(len(partTypes)):
            residuals.append({'partType': partTypes[i],
                              'partName': partNames[i]})

    residual_partTypes = [d['partType'] for d in residuals]
    residual_partNames = [d['partName'] for d in residuals]

    if len(residuals) > 0:
        kmeans_list = kmeans_list[:-1]
    update_dir = os.path.join(data_dir, 'update')
    key_indices_list, new_partNames, new_partTypes = computeDLFS(update_dir, mode)
    train_partNames = np.append(residual_partNames, new_partNames)
    train_partTypes = np.append(residual_partTypes, new_partTypes)

    last_id = meta[-1].get('id') + 1

    for partType in np.unique(new_partTypes):
        src = os.path.join(update_dir, partType)
        dest = os.path.join(data_dir, partType)
        shutil.copytree(src, dest, dirs_exist_ok=True)
        shutil.rmtree(src)
        os.mkdir(src)
        os.mkdir(src+'/STL')
        os.mkdir(src+'/STP')

    new_kmeans_list = get_kmeans_list(data_dir, train_partTypes, train_partNames, batch_size, num_clusters_batch)
    kmeans_list += new_kmeans_list
    return meta, kmeans_list, train_partTypes, train_partNames, new_partTypes, new_partNames


def delete():
    meta_path = os.path.join(data_dir, 'meta.json')
    delete_path = os.path.join(data_dir, 'delete')
    if not os.path.exists(meta_path):
        raise FileExistsError('meta.json does not exist!')
    else:
        with open(meta_path, 'r+') as file:
            meta = json.load(file)
            file.close()

    residuals = open_json(residuals_path)

    for partType in os.listdir(delete_path):
        if not os.path.isdir(os.path.join(delete_path, partType)):
            continue
        for partName in os.listdir(os.path.join(delete_path, partType, 'STL')):
            if not partName.lower().endswith('.stl'):
                continue
            file_name = partName[:-4]
            if os.path.exists(os.path.join(data_dir, partType, 'STL', file_name + '.stl')):
                os.remove(os.path.join(data_dir, partType, 'STL', file_name + '.stl'))
            if os.path.exists(os.path.join(data_dir, partType, 'DCT', file_name + '.npy')):
                os.remove(os.path.join(data_dir, partType, 'DCT', file_name + '.npy'))
            if os.path.exists(os.path.join(data_dir, partType, 'STP', file_name + '.stp')):
                os.remove(os.path.join(data_dir, partType, 'STP', file_name + '.stp'))
            if os.path.exists(os.path.join(data_dir, partType, 'STEP', file_name + '.stp')):
                os.remove(os.path.join(data_dir, partType, 'STEP', file_name + '.stp'))
            if os.path.exists(os.path.join(data_dir, partType, 'STP', file_name + '.step')):
                os.remove(os.path.join(data_dir, partType, 'STP', file_name + '.step'))
            if os.path.exists(os.path.join(data_dir, partType, 'STEP', file_name + '.step')):
                os.remove(os.path.join(data_dir, partType, 'STEP', file_name + '.step'))
            if os.path.exists(os.path.join(data_dir, partType, 'PCD', file_name + '.npy')):
                os.remove(os.path.join(data_dir, partType, 'PCD', file_name + '.npy'))
            i = 0
            while i < len(meta):
                if meta[i].get('partName') == file_name and meta[i].get('partType') == partType:
                    meta.pop(i)
                else:
                    i += 1
            i = 0
            while i < len(residuals):
                if residuals[i]['partName'] == file_name and residuals[i]['partType'] == partType:
                    residuals.pop(i)
                    print(file_name, 'removed')
                else:
                    i += 1

    for partType in os.listdir(delete_path):
        for partName in os.listdir(os.path.join(delete_path, partType, 'STL')):
            os.remove(os.path.join(delete_path, partType, 'STL', partName))

    id = 0
    for d in meta:
        d['id'] = id
        id += 1

    meta = json.dumps(meta, indent=2, ensure_ascii=False)
    residuals = json.dumps(residuals, indent=2, ensure_ascii=False)
    with open(meta_path, 'w') as outfile:
        outfile.write(meta)
        outfile.close()
    with open(residuals_path, 'w') as outfile:
        outfile.write(residuals)
        outfile.close()


if mode.lower() == 'temp_update':
    temp_update()
    sys.exit()

elif mode.lower() == "temp_update stp":
    temp_update()
    stp()
    sys.exit()

elif mode.lower().startswith('update'):
    meta, kmeans_list, train_partTypes, train_partNames, new_partTypes, new_partNames = update()

elif mode.lower() == 'delete':
    delete()
    sys.exit()

elif mode.lower() == 'stp':
    stp()
    sys.exit()

else:
    key_indices_list, new_partNames, new_partTypes = computeDLFS(data_dir, mode)
    batch_size = 500
    num_clusters_batch = 50
    train_partNames = new_partNames.copy()
    train_partTypes = new_partTypes.copy()
    kmeans_list = get_kmeans_list(data_dir, train_partTypes, train_partNames, batch_size, num_clusters_batch)

np.save(os.path.join(data_dir, 'kmeans_list.npy'), kmeans_list)
while len(train_partTypes) >= batch_size:
    train_partTypes = train_partTypes[batch_size:]
    train_partNames = train_partNames[batch_size:]

residuals = []
for i in range(len(train_partTypes)):
    residuals.append({'partType': train_partTypes[i],
                      'partName': train_partNames[i]})

with open(os.path.join(data_dir, 'residuals.json'), 'w') as outfile:
    residuals = json.dumps(residuals, indent=2, ensure_ascii=False)
    outfile.write(residuals)
    outfile.close()

if len(kmeans_list) < 2:
    high_kmeans = kmeans_list[0]
else:
    high_kmeans = construct_high_codebook(kmeans_list)

np.save(os.path.join(data_dir, 'high_kmeans.npy'), [high_kmeans])
print('Training finished in', (time.time() - tic0)/60, 'min. Initializing descriptor calculation.')

tic1 = time.time()
if mode.lower().startswith('update'):
    meta = update_meta_bof(meta, data_dir, high_kmeans, kmeans_list)
new_meta = computeGlobalDescriptors(data_dir, high_kmeans, kmeans_list, new_partTypes, new_partNames, last_id)
if meta:
    meta += new_meta
    meta = rm_dup(meta)
else:
    meta = new_meta

meta = json.dumps(meta, indent=2, ensure_ascii=False)
with open(os.path.join(data_dir, 'meta.json'), 'w') as outfile:
    outfile.write(meta)
    outfile.close()

print('BOF descriptor calculation finished in', (time.time() - tic1)/60, 'min.\n Total time consumed:', (time.time() - tic0)/60, 'min.')

if mode.endswith('stp'):
    stp()
