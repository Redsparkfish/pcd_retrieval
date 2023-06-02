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


file = open('configuration.json')
config = json.load(file)
data_dir = config['data_dir']
codebook = np.load(os.path.join(data_dir, 'codebook.npy'))
update_dir = os.path.join(data_dir, 'temp_update')

db = pymysql.connect(host=config['db_host'], user=config['db_user'], password=config['db_password'], database=config['db_database'])
cursor = db.cursor()
for category in os.listdir(update_dir):
    if not os.path.isdir(os.path.join(update_dir, category)):
        continue
    for filename in os.listdir(os.path.join(update_dir, category, 'STL')):
        if not filename.endswith('stl'):
            continue
        mesh = trimesh.load_mesh(os.path.join(update_dir, 'STL', filename))
        distribution_desc, bof_desc = calcDescriptors(mesh, codebook)

        data = {'partType': category,
                'partName': filename,
                'distributeDesc': numpy_to_json(distribution_desc),
                'bofDesc': numpy_to_json(bof_desc)}
        sql = "INSERT INTO `part_desc` (`partType`, `partName`, `distributeDesc`, `bofDesc`) VALUES (%s, %s, %s, %s) " \
              "AS new ON DUPLICATE KEY UPDATE " \
              "distributeDesc=new.distributeDesc, " \
              "bofDesc=new.bofDesc"
        cursor.execute(sql, (data['partType'], data['partName'], data['distributeDesc'], data['bofDesc']))
        print(filename, 'data uploaded')

db.commit()
db.close()
