from DLFS_calculation import computeDLFS
from bag_of_feature import *
from global_descriptor import *
import json

file = open('configuration.json', 'r')
config = json.load(file)
data_dir = config['data_dir']
DLFS_features, key_indices_list, names = computeDLFS(data_dir)
DLFS_features = np.array(DLFS_features)
bof = np.concatenate(DLFS_features, axis=0)
codebook, labels = construct_codebook(bof)
descriptors, D2s, names = computeGlobalDescriptors(data_dir, DLFS_features, codebook)

np.save(data_dir+'/names', names)
np.save(data_dir+'/DLFS_set', DLFS_features)
np.save(data_dir+'/codebook', codebook)
np.save(data_dir+'/bof_Desc', descriptors)
np.save(data_dir+'/D2_list', D2s)
