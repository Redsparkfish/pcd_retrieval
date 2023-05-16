import numpy as np


file = r'C:\Users\Admin\CAD_parts/DLFS_features.npy'
DLFS_set = np.load(file, allow_pickle=True)
size = DLFS_set.shape[0]
shape = DLFS_set[0].shape

con_set = []
for i in range(size):
    print(DLFS_set[i].reshape(shape[0], shape[1]*shape[2]).shape)
    con_set.append(DLFS_set[i].reshape(shape[0], shape[1]*shape[2]))

con_set = np.asarray(con_set)
bof = np.concatenate(con_set, axis=0)
np.save('C:/Users/Admin/CAD_parts/bof.npy', bof)
np.save('C:/Users/Admin/CAD_parts/con_DLFSs.npy', con_set)
