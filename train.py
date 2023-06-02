import shutil
from DLFS_calculation import computeDLFS
from bag_of_feature import *
from global_descriptor import *
import json

file = open('configuration.json', 'r')
config = json.load(file)
data_dir = config['data_dir']
update_dir = os.path.join(data_dir, 'update')
mode = config['mode']
if mode.lower() != 'update':
    DLFS_set, key_indices_list = computeDLFS(data_dir)
    DLFS_set = np.array(DLFS_set)
    bof = np.concatenate(DLFS_set, axis=0)
    codebook, labels = construct_codebook(bof)
    descriptors, D2s, names, categories = computeGlobalDescriptors(data_dir, DLFS_set, codebook)

    np.save(os.path.join(data_dir, 'names.npy'), names)
    np.save(os.path.join(data_dir, 'categories.npy'), categories)
    np.save(os.path.join(data_dir, 'DLFS_set.npy'), DLFS_set)
    np.save(os.path.join(data_dir, 'codebook.npy'), codebook)
    np.save(os.path.join(data_dir, 'bof_Desc.npy'), descriptors)
    np.save(os.path.join(data_dir, 'D2_list.npy'), D2s)
else:
    DLFS_set = np.load(os.path.join(data_dir, 'DLFS_set.npy'))
    bof = np.concatenate(DLFS_set, axis=0)

    new_DLFS_set, new_key_indices_list = computeDLFS(update_dir)
    new_DLFS_set = np.array(new_DLFS_set)
    new_bof = np.concatenate(new_DLFS_set, axis=0)

    bof = np.concatenate((bof, new_bof))
    codebook, labels = construct_codebook(bof)

    descriptors, D2s, names, categories = computeGlobalDescriptors(data_dir, DLFS_set, codebook)
    new_descriptors, new_D2s, new_names, new_categories = computeGlobalDescriptors(update_dir, new_DLFS_set, codebook)

    DLFS_set = np.concatenate((DLFS_set, new_DLFS_set))
    descriptors = np.concatenate((descriptors, new_descriptors))
    D2s = np.concatenate((D2s, new_D2s))
    names = np.concatenate((names, new_names))
    categories = np.concatenate((categories, new_categories))

    np.save(os.path.join(data_dir, 'names.npy'), names)
    np.save(os.path.join(data_dir, 'categories.npy'), categories)
    np.save(os.path.join(data_dir, 'DLFS_set.npy'), DLFS_set)
    np.save(os.path.join(data_dir, 'codebook.npy'), codebook)
    np.save(os.path.join(data_dir, 'bof_Desc.npy'), descriptors)
    np.save(os.path.join(data_dir, 'D2_list.npy'), D2s)

    for category in os.listdir(update_dir):
        src_dir = os.path.join(update_dir, category)
        tgt_dir = os.path.join(data_dir, category)
        if not os.path.isdir(src_dir):
            continue
        if not os.path.exists(tgt_dir):
            shutil.copytree(src_dir, tgt_dir)
        else:
            for file in os.listdir(os.path.join(src_dir, 'STL')):
                shutil.copy(os.path.join(src_dir, 'STL', file), os.path.join(tgt_dir, 'STL', file))
            for file in os.listdir(os.path.join(src_dir, 'PCD')):
                shutil.copy(os.path.join(src_dir, 'PCD', file), os.path.join(tgt_dir, 'PCD', file))
            for file in os.listdir(os.path.join(src_dir, 'STP')):
                shutil.copy(os.path.join(src_dir, 'STP', file), os.path.join(tgt_dir, 'STP', file))
        shutil.rmtree(os.path.join(src_dir, 'STL'))
        os.mkdir(os.path.join(src_dir, 'STL'))
        shutil.rmtree(os.path.join(src_dir, 'PCD'))
        os.mkdir(os.path.join(src_dir, 'PCD'))
        shutil.rmtree(os.path.join(src_dir, 'STP'))
        os.mkdir(os.path.join(src_dir, 'STP'))


