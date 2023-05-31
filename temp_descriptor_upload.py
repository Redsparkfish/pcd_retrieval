from global_descriptor import *
from DLFS_calculation import *
import pymysql
import json


def numpy_to_json(numpy_array):
    list_array = numpy_array.tolist()
    return json.dumps(list_array)


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


data_dir = r'C:\Users\Admin\CAD_parts'
codebook = np.load(r'C:\Users\Admin\CAD_parts/codebook.npy')
category = 'Bearings'
name = r'Bearings_115.stl'
mesh = trimesh.load_mesh(r'C:\Users\Admin\CAD_parts\Bearings\STL' + name)
distribution_desc, bof_desc = calcDescriptors(mesh, codebook)
db = pymysql.connect(host='106.15.224.125', user='demo', password='root1234!', database='demo')

data = {'partType': category,
        'partName': name,
        'distributeDesc': numpy_to_json(distribution_desc),
        'bofDesc': numpy_to_json(bof_desc)}
with db.cursor() as cursor:
    sql = "INSERT INTO `part_desc` (`partType`, `partName`, `distributeDesc`, `bofDesc`) VALUES (%s, %s, %s, %s)"
    cursor.execute(sql, (data['partType'], data['partName'], data['distributeDesc'], data['bofDesc']))

db.commit()
db.close()
