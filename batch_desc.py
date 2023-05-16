import os.path

import numpy as np
import trimesh

from DLFS_calculation import *
from global_descriptor import *


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


data_dir = 'C:/Users/Admin/CAD_parts'
# calculate d2 distributions
names = np.load('C:/Users/Admin/CAD_parts/names.npy')
N = len(names)
d2_list = np.zeros((N, 100))
i = 0
for category in os.listdir(data_dir):
    if not os.path.isdir(os.path.join(data_dir, category)):
        continue
    for file in os.listdir(os.path.join(data_dir, category, 'STL')):
        if not file.endswith('stl'):
            continue
        mesh = trimesh.load_mesh(os.path.join(data_dir, category, 'STL', file))
        points = trimesh.sample.sample_surface(mesh, 20000)[0]
        d2 = D2(points) / 20000
        d2_list[i] = d2
        i += 1
np.save('C:/Users/Admin/CAD_parts/d2_list', d2_list)
