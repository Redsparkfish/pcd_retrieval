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
names = np.load('C:/Users/Admin/CAD_parts/names.npy')
codebook = np.load('C:/Users/Admin/CAD_parts/codebook.npy')
N = len(names)
d2_list = np.zeros((N, 100))
bof_descriptors = np.zeros((N, 100))
i = 0
for category in os.listdir(data_dir):
    if not os.path.isdir(os.path.join(data_dir, category)):
        continue
    for file in os.listdir(os.path.join(data_dir, category, 'STL')):
        if not file.endswith('stl'):
            continue
        mesh = trimesh.load_mesh(os.path.join(data_dir, category, 'STL', file))
        d2, bof_desc = calcDescriptors(mesh, codebook)
        d2_list[i] = d2
        bof_descriptors[i] = bof_desc
        i += 1
np.save('C:/Users/Admin/CAD_parts/d2_list', d2_list)
np.save('C:/Users/Admin/CAD_parts/bof_desc', bof_descriptors)
