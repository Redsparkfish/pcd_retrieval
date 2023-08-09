import json
import os
import stat
import numpy as np


with open('configuration.json', 'r+', encoding='utf-8') as file:
    config = json.load(file)
    file.close()
data_dir = config['data_dir']


meta_path = os.path.join(data_dir, 'meta.json')
delete_path = os.path.join(data_dir, 'delete')
if not os.path.exists(meta_path):
    raise FileExistsError('meta.json does not exist!')
else:
    with open(meta_path, 'r+') as file:
        meta = json.load(file)
        file.close()

names_list = np.load(os.path.join(data_dir, 'names_list.npy'), allow_pickle=True).tolist()
categories_list = np.load(os.path.join(data_dir, 'categories_list.npy'), allow_pickle=True).tolist()

for partType in os.listdir(delete_path):
    for partName in os.listdir(os.path.join(delete_path, partType, 'STL')):
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
                print(len(meta))
            else:
                i += 1
        i = 0
        while i < len(names_list):
            j = 0
            while j < len(names_list[i]):
                if names_list[i][j] == file_name and categories_list[i][j] == partType:
                    names_list[i].pop(j)
                    categories_list[i].pop(j)
                    print(len(names_list[i]))
                    print(len(categories_list[i]))
                else:
                    j += 1
            i += 1

for partType in os.listdir(delete_path):
    for partName in os.listdir(os.path.join(delete_path, partType, 'STL')):
        os.remove(os.path.join(delete_path, partType, 'STL', partName))

id = 0
for d in meta:
    d['id'] = id
    id += 1

names_list = np.array(names_list, dtype='object')
categories_list = np.array(categories_list, dtype='object')
os.chmod(os.path.join(data_dir, 'names_list.npy'), stat.S_IWRITE)
os.chmod(os.path.join(data_dir, 'categories_list.npy'), stat.S_IWRITE)
os.chmod(meta_path, stat.S_IWRITE)
np.save(os.path.join(data_dir, 'names_list.npy'), names_list)
np.save(os.path.join(data_dir, 'categories_list.npy'), categories_list)
meta = json.dumps(meta)
with open(meta_path, 'w') as outfile:
    outfile.write(meta)
    outfile.close()


